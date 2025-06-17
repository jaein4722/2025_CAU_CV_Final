import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.03)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = F.silu(out)
        out = self.pointwise(out)
        return out

class OptimizedMultiDilationConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=False):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2 
        
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.03)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.03)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x1 = self.bn1(x1)
        x1 = F.silu(x1)

        x2 = self.depthwise2(x)
        x2 = self.bn2(x2)
        x2 = F.silu(x2)

        out = 0.65 * x1 + 0.35 * x2
        out = self.pointwise(out)
        return out

class OptimizedDownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels
        channels_conv = out_channels if not self.use_maxpool else out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.03)

    def forward(self, x):
        out = self.conv(x)
        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)
        out = self.bn(out)
        return F.silu(out)

class OptimizedResidualModule(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0, use_multi_dilation=False):
        super().__init__()
        if use_multi_dilation:
            self.conv = OptimizedMultiDilationConv2d(channels, channels, padding=1, dilation=dilation, bias=False)
        else:
            self.conv = OptimizedSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-4, momentum=0.03)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        
        out = 0.75 * identity + 0.25 * out
        return F.silu(out)

class OptimizedGradientModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self.refine_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.03)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.silu(out)

class OptimizedEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 채널 진행: 3 → 8 → 12 → 14
        
        self.downsample_1 = OptimizedDownsampleModule(in_channels, 8)
        self.gradient_module = OptimizedGradientModule(8)
        
        self.downsample_2 = OptimizedDownsampleModule(8, 12)
        self.residual_modules_2 = nn.Sequential(
            OptimizedResidualModule(12, dilation=1, dropout=0),
            OptimizedResidualModule(12, dilation=1, dropout=0)
        )
        
        self.downsample_3 = OptimizedDownsampleModule(12, 14)
        
        # 2개 MultiDilation 모듈
        rates = [4, 8]
        self.feature_modules = nn.Sequential(*[
            OptimizedResidualModule(14, dilation=rate, dropout=0.03, use_multi_dilation=True)
            for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        g_feat = self.gradient_module(d1)
        d1_enhanced = 0.85 * d1 + 0.15 * g_feat
        
        d2 = self.downsample_2(d1_enhanced)
        m2 = self.residual_modules_2(d2)
        
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2

class OptimizedUpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.03)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.silu(out)

class submission_MiniNetV8(nn.Module):
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        
        self.encoder = OptimizedEncoder(in_channels)
        
        self.aux_downsample = OptimizedDownsampleModule(in_channels, 8)
        self.aux_refine = OptimizedResidualModule(8, dilation=1, dropout=0)
        
        self.upsample_1 = OptimizedUpsampleModule(14, 12)
        self.upsample_mods = nn.Sequential(
            OptimizedResidualModule(12, dilation=1, dropout=0),
            OptimizedResidualModule(12, dilation=1, dropout=0)
        )
        
        self.output_conv = nn.ConvTranspose2d(12, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        enc, skip = self.encoder(x)
        
        up1 = self.upsample_1(enc)
        
        if up1.shape[2:] == skip.shape[2:]:
            up1 = 0.7 * up1 + 0.3 * skip
        if up1.shape[2:] == aux.shape[2:]:
            up1 = 0.9 * up1 + 0.1 * aux
            
        m3 = self.upsample_mods(up1)
        out = self.output_conv(m3)
        
        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return out

if __name__ == "__main__":
    print("=== MiniNetV8 파라미터 분석 ===")
    
    net_voc = submission_MiniNetV8(in_channels=3, num_classes=21)
    net_binary = submission_MiniNetV8(in_channels=3, num_classes=2)
    
    params_voc = sum(p.numel() for p in net_voc.parameters() if p.requires_grad)
    params_binary = sum(p.numel() for p in net_binary.parameters() if p.requires_grad)
    
    print(f"VOC (21 classes): {params_voc:,}")
    print(f"Binary (2 classes): {params_binary:,}")
    print(f"Binary 목표: ~7,000")
    print(f"Binary 상태: {'✅ OK' if 6500 <= params_binary <= 7500 else '❌ ADJUST'}")
    
    x = torch.randn(1, 3, 256, 256)
    y_voc = net_voc(x)
    y_binary = net_binary(x)
    print(f"VOC output: {y_voc.shape}")
    print(f"Binary output: {y_binary.shape}")
    
    print("\n=== 최적화 기법 ===")
    print("1. SiLU 활성화 (빠르고 효과적)")
    print("2. 채널 최적화 (3→8→12→14)")
    print("3. MultiDilation 2개 집중 (4,8)")
    print("4. 정교한 가중 결합") 