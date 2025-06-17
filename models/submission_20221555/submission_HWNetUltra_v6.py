import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules (v5와 동일) ---

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

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn, self.nOut = nIn, nOut
        nConv = nOut - nIn if self.nIn < self.nOut else nOut
        self.conv3x3 = DWSConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            output = torch.cat([output, self.max_pool(input)], 1)
        return self.acti(self.bn(output))

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
    def __init__(self, nIn, d=1, p=0.25):
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

class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(4, out_channels // 4)
        
        self.branch1 = DWSConv(in_channels, mid_channels, 1, 1, padding=0, bn_acti=True)
        self.branch2 = DWSConv(in_channels, mid_channels, 3, 1, padding=6, dilation=(6, 6), bn_acti=True)
        self.branch3 = DWSConv(in_channels, mid_channels, 3, 1, padding=12, dilation=(12, 12), bn_acti=True)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1e-3),
            nn.SiLU(inplace=True)
        )
        
        self.fusion = DWSConv(mid_channels * 4, out_channels, 1, 1, padding=0, bn_acti=True)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        concat = torch.cat([b1, b2, b3, gp], dim=1)
        return self.fusion(concat)

# --- v6 새로운 모듈들 ---

class CoordinateAttention(nn.Module):
    """Coordinate Attention - 공간 정보 보존"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(1, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return identity * a_h * a_w

class EnhancedSkipConnection(nn.Module):
    """Enhanced Skip Connection with learnable weights"""
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        self.align_high = nn.Conv2d(high_channels, out_channels, 1, bias=False) if high_channels != out_channels else nn.Identity()
        self.align_low = nn.Conv2d(low_channels, out_channels, 1, bias=False) if low_channels != out_channels else nn.Identity()
        
        self.weight_high = nn.Parameter(torch.ones(1))
        self.weight_low = nn.Parameter(torch.ones(1))
        
        self.refine = DWSConv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        
    def forward(self, high_feat, low_feat):
        high_up = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        
        high_aligned = self.align_high(high_up)
        low_aligned = self.align_low(low_feat)
        
        fused = self.weight_high * high_aligned + self.weight_low * low_aligned
        
        weight_sum = torch.abs(self.weight_high) + torch.abs(self.weight_low)
        fused = fused * 2.0 / (weight_sum + 1e-8)
        
        return self.refine(fused)

class EdgeEnhancedOutput(nn.Module):
    """Edge-enhanced output module"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.main_conv = nn.Sequential(
            DWSConv(in_channels + 1, in_channels//2, 3, 1, padding=1, bn_acti=True),
            nn.Conv2d(in_channels//2, num_classes, 1)
        )
        
    def forward(self, x):
        edge = self.edge_conv(x)
        x_enhanced = torch.cat([x, edge], dim=1)
        return self.main_conv(x_enhanced)

# --- 최종 모델 ---
class submission_HWNetUltra_v6(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        block_1 = 2
        block_2 = 2
        C = 8
        P = 0.25
        dilation_block_1 = [2, 3]
        dilation_block_2 = [2, 4]

        self.edge_focus = MorphGradientFocus(in_channels)
        
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        self.aspp = LightASPP(C * 4, C * 4)
        self.coord_attention = CoordinateAttention(C * 4, reduction=8)
        
        self.skip2 = EnhancedSkipConnection(C * 4, C * 2, C * 2)
        self.skip1 = EnhancedSkipConnection(C * 2, C, C)
        
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output = EdgeEnhancedOutput(C, num_classes)

    def forward(self, input):
        x_init = self.edge_focus(input)
        
        x0 = self.Init_Block(x_init)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        
        x2 = self.aspp(x2)
        x2 = self.coord_attention(x2)
        
        d2 = self.skip2(x2, x1)
        d1 = self.skip1(d2, x0)
        
        d1 = self.final_up(d1)
        return self.output(d1)

if __name__ == "__main__":
    for num_classes in [2, 21]:
        net = submission_HWNetUltra_v6(in_channels=3, num_classes=num_classes)
        p = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Classes: {num_classes}, Params: {p:,}")
        
        try:
            net.eval()
            x = torch.randn(1, 3, 256, 256)
            y = net(x)
            print(f"✅ Output: {y.shape}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 HWNetUltra_v6:")
    print(f"  🆕 Coordinate Attention: 공간 정보 보존")
    print(f"  🆕 Enhanced Skip: 학습 가능한 fusion")  
    print(f"  🆕 Edge-Enhanced Output: 경계 강화")
    print(f"  목표: IoU 0.4051 → 0.43+")
