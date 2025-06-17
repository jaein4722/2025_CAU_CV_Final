import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MiniNetV5: 파라미터 수 유지하며 MiniNetV3 성능 재현
# 전략: 구조 최적화 + 효율적 파라미터 재분배
# =============================================================================

class OptimizedSeparableConv2d(nn.Module):
    """최적화된 Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pointwise(out)
        return out

class OptimizedMultiDilationConv2d(nn.Module):
    """MiniNetV3 스타일 MultiDilation - 파라미터 효율성 극대화"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=False):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2 
        
        # MiniNetV3와 동일한 구조 - 각 branch별 BN
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
        # MiniNetV3와 동일 - 각 branch별 BN
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        # MiniNetV3와 정확히 동일한 forward
        x1 = self.depthwise1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.depthwise2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        out = x1 + x2  # 두 branch 합산
        out = self.pointwise(out)
        return out

class OptimizedDownsampleModule(nn.Module):
    """MiniNetV3와 동일한 다운샘플링"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels
        channels_conv = out_channels if not self.use_maxpool else out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)
        out = self.bn(out)
        return F.relu(out)

class OptimizedResidualModule(nn.Module):
    """최적화된 Residual 모듈"""
    def __init__(self, channels, dilation=1, dropout=0, use_multi_dilation=False):
        super().__init__()
        if use_multi_dilation:
            self.conv = OptimizedMultiDilationConv2d(channels, channels, padding=1, dilation=dilation, bias=False)
        else:
            self.conv = OptimizedSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class OptimizedGradientModule(nn.Module):
    """MiniNetV3와 동일한 Gradient Module"""
    def __init__(self, in_channels):
        super().__init__()
        # Sobel 필터 (파라미터 없음)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # MiniNetV3와 동일한 refinement
        self.refine_conv = OptimizedSeparableConv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.relu(out)

class OptimizedEncoder(nn.Module):
    """파라미터 효율적 인코더 - MiniNetV3 구조 재현"""
    def __init__(self, in_channels):
        super().__init__()
        # 채널 진행: 3 → 8 → 14 → 18 (파라미터 절약)
        
        self.downsample_1 = OptimizedDownsampleModule(in_channels, 8)
        self.gradient_module = OptimizedGradientModule(8)
        
        self.downsample_2 = OptimizedDownsampleModule(8, 14)  # 16→14 축소
        self.residual_modules_2 = nn.Sequential(
            OptimizedResidualModule(14, dilation=1, dropout=0),
            OptimizedResidualModule(14, dilation=1, dropout=0)
        )
        
        self.downsample_3 = OptimizedDownsampleModule(14, 16)  # 18→16 추가 축소
        
        # 3개 MultiDilation 모듈 (4개→3개 축소하되 핵심 dilation 유지)
        rates = [2, 4, 8]  # 1 제거, 핵심 dilation만 유지
        self.feature_modules = nn.Sequential(*[
            OptimizedResidualModule(16, dilation=rate, dropout=0.1, use_multi_dilation=True) 
            for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        g_feat = self.gradient_module(d1)
        d1_enhanced = d1 + g_feat
        
        d2 = self.downsample_2(d1_enhanced)
        m2 = self.residual_modules_2(d2)
        
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2

class OptimizedUpsampleModule(nn.Module):
    """MiniNetV3 스타일 ConvTranspose2d 사용"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # MiniNetV3와 동일한 ConvTranspose2d 사용
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)

class submission_MiniNetV5(nn.Module):
    """파라미터 수 유지하며 MiniNetV3 성능 재현"""
    
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        
        # 메인 인코더
        self.encoder = OptimizedEncoder(in_channels)
        
        # Auxiliary path
        self.aux_downsample = OptimizedDownsampleModule(in_channels, 8)
        self.aux_refine = OptimizedResidualModule(8, dilation=1, dropout=0)
        
        # 디코더 - 파라미터 절약을 위해 채널 수 조정
        self.upsample_1 = OptimizedUpsampleModule(16, 14)  # 16→14
        self.upsample_mods = nn.Sequential(
            OptimizedResidualModule(14, dilation=1, dropout=0),
            OptimizedResidualModule(14, dilation=1, dropout=0)
        )
        
        # 최종 출력 - MiniNetV3 스타일
        self.output_conv = nn.ConvTranspose2d(14, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        
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
        # Auxiliary path
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # Decoder
        up1 = self.upsample_1(enc)
        
        # Skip connections
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        if up1.shape[2:] == aux.shape[2:]:
            up1 = up1 + aux
            
        # Final processing
        m3 = self.upsample_mods(up1)
        out = self.output_conv(m3)
        
        # 원본 크기로 복원
        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        return out

if __name__ == "__main__":
    # 테스트
    net = submission_MiniNetV5(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {params:,}")
    print(f"MiniNetV4 대비: {params - 9463:+,} params")
    print(f"10k limit: {'✅ PASS' if params <= 10000 else '❌ FAIL'}")
    
    # 모듈별 파라미터 분석
    print("\n=== Module Analysis ===")
    for name, module in net.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {module_params:,} ({module_params/params*100:.1f}%)") 