import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules ---

class DWSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, dilation=(1, 1), bn_acti=True, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding,
                                   dilation=dilation, groups=nIn, bias=bias)
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_acti = bn_acti
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_acti:
            x = self.bn(x)
            x = self.acti(x)
        return x

class Conv(nn.Module):
    """Standard or Grouped Convolution"""
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn(output)
            output = self.acti(output)
        return output

# [전략 2] 더 표준적이고 효율적인 다운샘플링 블록
class SimplifiedDownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = DWSConv(nIn, nOut, kSize=3, stride=2, padding=1, bn_acti=True)

    def forward(self, input):
        return self.conv(input)

def Split(x, p):
    c = int(x.size(1))
    c1 = round(c * (1 - p))
    return x[:, :c1, :, :].contiguous(), x[:, c1:, :, :].contiguous()

class TCA(nn.Module):
    def __init__(self, c, d=1, kSize=3):
        super().__init__()
        self.conv3x3 = DWSConv(c, c, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x3 = Conv(c, c, (kSize, kSize), 1, padding=(1, 1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c, c, (kSize, kSize), 1, padding=(d, d), groups=c, dilation=(d, d), bn_acti=True)
        self.bn = nn.BatchNorm2d(c, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        br = self.conv3x3(input)
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        return self.acti(self.bn(br))

class PCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = nIn - round(nIn * (1 - p))
        self.TCA = TCA(c, d)
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        x1, x2 = Split(input, self.p)
        x2 = self.TCA(x2)
        output = torch.cat([x1, x2], dim=1)
        return self.conv1x1(output) + input

class LightDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DWSConv(in_channels + skip_channels, out_channels, 3, 1, padding=1, bn_acti=True),
            DWSConv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class MorphGradientFocus(nn.Module):
    def __init__(self, in_channels, k=3):
        super().__init__()
        self.pad  = k // 2
        self.fuse = Conv(in_channels + 1, in_channels, 1, 1, padding=0, bn_acti=True)

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)
        dilated = F.max_pool2d(intensity, 3, stride=1, padding=self.pad)
        eroded  = -F.max_pool2d(-intensity, 3, stride=1, padding=self.pad)
        return self.fuse(torch.cat([x, dilated - eroded], dim=1))

# --- 최종 제출 모델 ---
class submission_HWNetPico_v1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # [전략 1] 너비를 8에서 10으로 소폭 증가
        C = 10
        # 깊이는 v2와 동일하게 유지
        block_1 = 2
        block_2 = 2
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [4, 4]

        self.edge_focus = MorphGradientFocus(in_channels)
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        
        # [전략 2] 다운샘플링 블록을 단순화된 버전으로 교체
        self.down1 = SimplifiedDownSamplingBlock(C, C * 2)
        self.LC_Block_1 = nn.Sequential()
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        self.down2 = SimplifiedDownSamplingBlock(C * 2, C * 4)
        self.LC_Block_2 = nn.Sequential()
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        self.dec2 = LightDecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = LightDecoderBlock(C * 2, C, C)
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True),
            nn.Conv2d(C, num_classes, kernel_size=1)
        )

    def forward(self, input):
        x_init = self.edge_focus(input)
        x0 = self.Init_Block(x_init)
        
        x1_down = self.down1(x0)
        x1 = self.LC_Block_1(x1_down)
        
        x2_down = self.down2(x1)
        x2 = self.LC_Block_2(x2_down)

        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        return self.final_up(d1)

if __name__ == "__main__":
    num_classes = 21
    net = submission_HWNetPico_v1(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNet-Pico_v1")
    print(f"Trainable Params: {p/1e3:.2f} K") # 목표: < 10 K

    try:
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("Test Passed: Model runs successfully and output shape is correct.")
    except Exception as e:
        print(f"Test Failed: {e}")