import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────── 공용 블록 (안정적인 ReLU + BN 기반) ──────────────────────
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, input):
        return self.acti(self.bn(self.conv(input)))

class DSConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding,
                            dilation=dilation, groups=nIn, bias=bias)
        self.pw = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, input):
        return self.acti(self.bn(self.pw(self.dw(input))))

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn if nIn < nOut else nOut
        self.conv3x3 = DSConv(nIn, nConv, 3, 2, 1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.ReLU(inplace=True)

    def forward(self, input):
        x_conv = self.conv3x3(input)
        if input.size(1) < self.nOut:
            x_pool = self.max_pool(input)
            x_conv = torch.cat([x_conv, x_pool], 1)
        return self.acti(self.bn(x_conv))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DSConv(in_channels // 2 + skip_channels, out_channels, 3, 1, 1),
            DSConv(out_channels, out_channels, 3, 1, 1)
        )
    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class MorphGradientFocus(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuse = Conv(in_channels + 1, in_channels, 1, 1, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = F.max_pool2d(x.mean(1,True),3,1,1) - (-F.max_pool2d(-x.mean(1,True),3,1,1))
        return self.fuse(torch.cat([x, edge], dim=1))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ★★★★★ v6를 위한 초경량 블록 ★★★★★
class TCA_Ultralight(nn.Module):
    """[개선점 2] 단일 Dilated DSConv 경로"""
    def __init__(self, c, d=1):
        super().__init__()
        self.conv = DSConv(c, c, kSize=3, stride=1, padding=d, dilation=d)
    def forward(self, input):
        return self.conv(input)

class PCT_Ultralight(nn.Module):
    """[개선점 1] 비싼 1x1 Conv를 제거하고 순수 잔차 연결로 변경"""
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c_split = int(nIn * p)
        self.tca = TCA_Ultralight(c_split, d)

    def forward(self, input):
        c_split = int(input.size(1) * self.p)
        x_tca, x_pass = input[:, :c_split], input[:, c_split:]
        x_tca = self.tca(x_tca)
        output = torch.cat([x_tca, x_pass], dim=1)
        return output + input # 1x1 Conv 대신 Residual Connection

# ────────────────────── HWNetv6_Ultralight ──────────────────────
class submission_HWNetv5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # 파라미터 1만 초반을 위한 설정
        block_1 = 2 
        block_2 = 4
        C = 8       # [개선점 3]
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [2, 4, 4, 8]

        self.edge_focus = MorphGradientFocus(in_channels)
        self.Init_Block = nn.Sequential(
            DSConv(in_channels, C, 3, 2, 1),
            DSConv(C, C, 3, 1, 1)
        )
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT_Ultralight(C * 2, d=dilation_block_1[i], p=P))
            
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT_Ultralight(C * 4, d=dilation_block_2[i], p=P))
            
        self.attn = SEBlock(C * 4) # 성능 보존을 위해 유지
        
        self.dec2 = DecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = DecoderBlock(C * 2, C, C)
        
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Conv2d(C, num_classes, kernel_size=1)

    def forward(self, input):
        input_focused = self.edge_focus(input)
        x0 = self.Init_Block(input_focused)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        x2_attn = self.attn(x2)
        d2 = self.dec2(x2_attn, x1)
        d1 = self.dec1(d2, x0)
        d1_dropped = self.dropout(d1)
        out = F.interpolate(d1_dropped, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.head(out)
        return out

if __name__ == "__main__":
    net = submission_HWNetv5(in_channels=3, num_classes=21)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable Params: {p/1e3:.1f} K")