import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MiniNetV4_Compact: MiniNetV3의 성능 유지하며 10k 파라미터 달성
# 핵심 전략: 채널 수 축소 + 효율적 모듈 재설계
# =============================================================================

class CompactSeparableConv2d(nn.Module):
    """경량화된 Separable Convolution"""
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

class CompactMultiDilationSeparableConv2d(nn.Module):
    """MiniNetV3의 핵심 기술 - 파라미터 최적화"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=False):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2 
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)
        out = x1 + x2  # 두 dilation 결과 합산
        out = self.pointwise(out)
        out = self.bn(out)
        return F.relu(out)

class CompactDownsampleModule(nn.Module):
    """경량화된 다운샘플링 - 채널 수 축소"""
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

class CompactResidualModule(nn.Module):
    """경량화된 Residual 모듈"""
    def __init__(self, channels, dilation=1, dropout=0, use_multi_dilation=False):
        super().__init__()
        if use_multi_dilation:
            self.conv = CompactMultiDilationSeparableConv2d(channels, channels, padding=1, dilation=dilation, bias=False)
        else:
            self.conv = CompactSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class CompactGradientModule(nn.Module):
    """경량화된 Gradient Feature Module - MiniNetV3의 핵심"""
    def __init__(self, in_channels):
        super().__init__()
        # Sobel 필터 (학습되지 않는 파라미터)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 경량화된 refinement (파라미터 최소화)
        self.refine_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.relu(out)

class CompactEncoder(nn.Module):
    """경량화된 인코더 - 채널 수 대폭 축소"""
    def __init__(self, in_channels):
        super().__init__()
        # 채널 진행: 3 → 8 → 16 → 20 (기존 3 → 10 → 20 → 26에서 축소)
        
        self.downsample_1 = CompactDownsampleModule(in_channels, 8)  # 10 → 8
        self.gradient_module = CompactGradientModule(8)
        
        self.downsample_2 = CompactDownsampleModule(8, 16)  # 20 → 16
        self.residual_modules_2 = nn.Sequential(
            CompactResidualModule(16, dilation=1, dropout=0),
            CompactResidualModule(16, dilation=1, dropout=0)
        )
        
        self.downsample_3 = CompactDownsampleModule(16, 20)  # 26 → 20
        
        # Multi-dilation 모듈 수 축소 (4개 → 2개)
        self.feature_modules = nn.Sequential(
            CompactResidualModule(20, dilation=2, dropout=0.1, use_multi_dilation=True),
            CompactResidualModule(20, dilation=4, dropout=0.1, use_multi_dilation=True)
        )

    def forward(self, x):
        d1 = self.downsample_1(x)
        g_feat = self.gradient_module(d1)
        d1_enhanced = d1 + g_feat
        
        d2 = self.downsample_2(d1_enhanced)
        m2 = self.residual_modules_2(d2)
        
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection용

class CompactUpsampleModule(nn.Module):
    """경량화된 업샘플링 모듈"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d 대신 Interpolation + Conv 사용 (파라미터 절약)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        # 2배 업샘플링
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)

class submission_MiniNetV4(nn.Module):
    """MiniNetV3 성능 유지하며 10k 파라미터 달성"""
    
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        
        # 메인 인코더
        self.encoder = CompactEncoder(in_channels)
        
        # Auxiliary path (경량화)
        self.aux_downsample = CompactDownsampleModule(in_channels, 8)  # 10 → 8
        self.aux_refine = CompactResidualModule(8, dilation=1, dropout=0)
        
        # 디코더 (경량화)
        self.upsample_1 = CompactUpsampleModule(20, 16)  # 26→20, 20→16
        self.upsample_mods = nn.Sequential(
            CompactResidualModule(16, dilation=1, dropout=0),
            CompactResidualModule(16, dilation=1, dropout=0)
        )
        
        # 최종 출력 (파라미터 최소화)
        self.output_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1, bias=True)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
        
        # Skip connections (크기 맞춤)
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
    net = submission_MiniNetV4(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {params:,}")
    print(f"10k limit: {'✅ PASS' if params <= 10000 else '❌ FAIL'}")
    
    # 모듈별 파라미터 분석
    print("\n=== Module Analysis ===")
    for name, module in net.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {module_params:,} ({module_params/params*100:.1f}%)") 