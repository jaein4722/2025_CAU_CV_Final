import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────── 1. 안정화된 기본 블록 (ReLU + BN) ──────────────────────
# 이 파일 하나로 실행될 수 있도록 모든 클래스를 포함합니다.

class ConvReLU(nn.Module):
    """표준 Conv + BN + ReLU 블록"""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, input):
        return self.acti(self.bn(self.conv(input)))

class DSConvReLU(nn.Module):
    """표준 DSConv + BN + ReLU 블록"""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(nIn, nIn, kSize, stride, padding, dilation=dilation, groups=nIn, bias=bias)
        self.pw = nn.Conv2d(nIn, nOut, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, input):
        return self.acti(self.bn(self.pw(self.dw(input))))

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        nConv = nOut - nIn
        self.conv_branch = DSConvReLU(nIn, nConv, 3, 2, 1)
        self.pool_branch = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        x_conv = self.conv_branch(input)
        x_pool = self.pool_branch(input)
        return torch.cat([x_conv, x_pool], 1)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DSConvReLU(out_channels + skip_channels, out_channels, 3, 1, 1),
            DSConvReLU(out_channels, out_channels, 3, 1, 1)
        )
    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ★★★★★ 이전 코드에서 누락되었던 클래스 ★★★★★
class MorphGradientFocus(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 내부 Conv도 안정화된 ConvReLU를 사용하도록 수정
        self.fuse = ConvReLU(in_channels + 1, in_channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = F.max_pool2d(x.mean(1,True),3,1,1) - (-F.max_pool2d(-x.mean(1,True),3,1,1))
        return self.fuse(torch.cat([x, edge], dim=1))

# ────────────────────── 2. 경량화된 핵심 블록 ──────────────────────
class PCT_Slim(nn.Module):
    """TCA와 1x1 Conv를 모두 제거하고, 단일 Dilated DSConv와 잔차 연결만 사용"""
    def __init__(self, C, d, p=0.5):
        super().__init__()
        self.p = p
        split_channels = int(C * p)
        self.tca_light = DSConvReLU(split_channels, split_channels, 3, 1, padding=d, dilation=d)
    
    def forward(self, x):
        split_idx = int(x.size(1) * self.p)
        x1, x2 = x[:, :split_idx], x[:, split_idx:]
        x1_transformed = self.tca_light(x1)
        x_out = torch.cat([x1_transformed, x2], 1)
        # 입력과 출력의 채널 수가 같으므로 잔차 연결 가능
        return x_out + x

# ────────────────────── 3. HWNet-v6 (최종 설계) ──────────────────────
class submission_HWNetv6(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # 파라미터 1.4만개를 위한 설정
        C = 8
        block_1_repeats = 2
        block_2_repeats = 6
        P = 0.5
        dilation_1 = [2,2]
        dilation_2 = [4,4,8,8,16,16]

        self.edge_focus = MorphGradientFocus(in_channels)
        self.stem = nn.Sequential(
            DSConvReLU(in_channels, C, 3, 2, 1),
            DSConvReLU(C, C, 3, 1, 1)
        )

        self.encoder_s1 = nn.Sequential()
        self.encoder_s1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1_repeats):
            self.encoder_s1.add_module(f"pct_{i}", PCT_Slim(C * 2, d=dilation_1[i], p=P))

        self.encoder_s2 = nn.Sequential()
        self.encoder_s2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2_repeats):
            self.encoder_s2.add_module(f"pct_{i}", PCT_Slim(C * 4, d=dilation_2[i], p=P))
        
        self.bottleneck_attn = SEBlock(C * 4)
        
        self.decoder_s2 = DecoderBlock(C*4, C*2, C*2)
        self.decoder_s1 = DecoderBlock(C*2, C, C)
        
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(C, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x_focused = self.edge_focus(x)
        x0 = self.stem(x_focused)
        x1 = self.encoder_s1(x0)
        x2 = self.encoder_s2(x1)
        
        b = self.bottleneck_attn(x2)
        
        d1 = self.decoder_s2(b, x1)
        d0 = self.decoder_s1(d1, x0)
        
        out = self.head(d0)
        return F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

if __name__ == "__main__":
    net = submission_HWNetv6(in_channels=3, num_classes=2)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable Params: {p/1e3:.1f} K")