import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules ---

class DWSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, bn_acti=True):
        super().__init__()
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding, groups=nIn, bias=False)
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_acti = bn_acti
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_acti: return self.acti(self.bn(x))
        return x

class LightDecoderBlock(nn.Module):
    """Parameter-free upsampling Decoder"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DWSConv(in_channels + skip_channels, out_channels, 3, 1, padding=1, bn_acti=True),
            DWSConv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        )
    def forward(self, x, skip_x):
        x = self.up(x)
        return self.conv(torch.cat([x, skip_x], dim=1))

# --- 최종 제출 모델: HWNet-Plain_v1 ---
class submission_HWNetPlain_v1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        C = 12 # 기본 채널 수
        
        # 1. Stem: DWSConv 기반의 극도로 가벼운 스템
        self.stem = DWSConv(in_channels, C, kSize=3, stride=2, padding=1)
        
        # 2. Stage 1: 단순 DWSConv 스택
        self.stage1 = nn.Sequential(
            DWSConv(C, C, kSize=3, stride=1, padding=1),
            DWSConv(C, C, kSize=3, stride=1, padding=1),
        )
        self.down1 = DWSConv(C, C * 2, kSize=3, stride=2, padding=1)
        
        # 3. Stage 2
        self.stage2 = nn.Sequential(
            DWSConv(C * 2, C * 2, kSize=3, stride=1, padding=1),
            DWSConv(C * 2, C * 2, kSize=3, stride=1, padding=1),
        )
        self.down2 = DWSConv(C * 2, C * 4, kSize=3, stride=2, padding=1)

        # 4. Stage 3
        self.stage3 = nn.Sequential(
            DWSConv(C * 4, C * 4, kSize=3, stride=1, padding=1),
            DWSConv(C * 4, C * 4, kSize=3, stride=1, padding=1),
        )
        
        # 5. Decoder
        self.dec2 = LightDecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = LightDecoderBlock(C * 2, C, C)
        
        # 6. Final Layer
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(C, num_classes, kernel_size=1)
        )

    def forward(self, input):
        s1_out = self.stage1(self.stem(input))
        s2_out = self.stage2(self.down1(s1_out))
        s3_out = self.stage3(self.down2(s2_out))

        d2 = self.dec2(s3_out, s2_out)
        d1 = self.dec1(d2, s1_out)
        
        return self.final_conv(d1)

if __name__ == "__main__":
    num_classes = 1
    net = submission_HWNetPlain_v1(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNet-Plain_v1 (Final)")
    print(f"Trainable Params: {p/1e3:.2f} K")

    try:
        x = torch.randn(2, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (2, num_classes, 256, 256)
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")