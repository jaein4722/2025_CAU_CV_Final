import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MiniNetV7: MiniNetV5와 동일 구조 + 파라미터 제로 증가 최적화
# 전략: 구조 동일 유지 + 활성화 함수/초기화/정규화 최적화
# =============================================================================

class ZeroParamSeparableConv2d(nn.Module):
    """MiniNetV5와 동일 구조 + 개선된 설정"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)  # 더 안정적인 BN

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = F.gelu(out)  # ReLU → GELU (성능 향상, 파라미터 증가 없음)
        out = self.pointwise(out)
        return out

class ZeroParamMultiDilationConv2d(nn.Module):
    """MiniNetV5와 동일 구조 + 개선된 forward"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2, bias=False):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2 
        
        # MiniNetV5와 완전 동일한 구조
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
        # MiniNetV5와 동일한 BN 구조
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        # MiniNetV5와 거의 동일하되 활성화 함수만 개선
        x1 = self.depthwise1(x)
        x1 = self.bn1(x1)
        x1 = F.gelu(x1)  # ReLU → GELU

        x2 = self.depthwise2(x)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2)  # ReLU → GELU

        # 가중 합산으로 성능 향상
        out = 0.6 * x1 + 0.4 * x2  # 균등 합산 → 가중 합산
        out = self.pointwise(out)
        return out

class ZeroParamDownsampleModule(nn.Module):
    """MiniNetV5와 완전 동일"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels
        channels_conv = out_channels if not self.use_maxpool else out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        out = self.conv(x)
        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)
        out = self.bn(out)
        return F.gelu(out)  # ReLU → GELU

class ZeroParamResidualModule(nn.Module):
    """MiniNetV5와 동일 구조 + 개선된 residual connection"""
    def __init__(self, channels, dilation=1, dropout=0, use_multi_dilation=False):
        super().__init__()
        if use_multi_dilation:
            self.conv = ZeroParamMultiDilationConv2d(channels, channels, padding=1, dilation=dilation, bias=False)
        else:
            self.conv = ZeroParamSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        
        # 향상된 residual connection (파라미터 증가 없음)
        out = 0.7 * identity + 0.3 * out  # 가중 residual
        return F.gelu(out)

class ZeroParamGradientModule(nn.Module):
    """MiniNetV5와 완전 동일 구조"""
    def __init__(self, in_channels):
        super().__init__()
        # MiniNetV5와 동일한 Sobel 필터
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # MiniNetV5와 동일한 refinement
        self.refine_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.gelu(out)  # ReLU → GELU

class ZeroParamEncoder(nn.Module):
    """MiniNetV5와 완전 동일한 구조"""
    def __init__(self, in_channels):
        super().__init__()
        # MiniNetV5와 정확히 동일한 채널 구조: 3 → 8 → 14 → 16
        
        self.downsample_1 = ZeroParamDownsampleModule(in_channels, 8)
        self.gradient_module = ZeroParamGradientModule(8)
        
        self.downsample_2 = ZeroParamDownsampleModule(8, 14)
        self.residual_modules_2 = nn.Sequential(
            ZeroParamResidualModule(14, dilation=1, dropout=0),
            ZeroParamResidualModule(14, dilation=1, dropout=0)
        )
        
        self.downsample_3 = ZeroParamDownsampleModule(14, 16)
        
        # MiniNetV5와 동일 - 3개 MultiDilation 모듈
        rates = [2, 4, 8]
        self.feature_modules = nn.Sequential(*[
            ZeroParamResidualModule(16, dilation=rate, dropout=0.05, use_multi_dilation=True)  # dropout 약간 감소
            for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        g_feat = self.gradient_module(d1)
        d1_enhanced = 0.8 * d1 + 0.2 * g_feat  # 가중 결합으로 성능 향상
        
        d2 = self.downsample_2(d1_enhanced)
        m2 = self.residual_modules_2(d2)
        
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2

class ZeroParamUpsampleModule(nn.Module):
    """MiniNetV5와 완전 동일"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.gelu(out)  # ReLU → GELU

class submission_MiniNetV7(nn.Module):
    """MiniNetV5와 동일 구조 + 파라미터 제로 증가 최적화"""
    
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        
        # MiniNetV5와 완전 동일한 구조
        self.encoder = ZeroParamEncoder(in_channels)
        
        # MiniNetV5와 동일
        self.aux_downsample = ZeroParamDownsampleModule(in_channels, 8)
        self.aux_refine = ZeroParamResidualModule(8, dilation=1, dropout=0)
        
        # MiniNetV5와 동일
        self.upsample_1 = ZeroParamUpsampleModule(16, 14)
        self.upsample_mods = nn.Sequential(
            ZeroParamResidualModule(14, dilation=1, dropout=0),
            ZeroParamResidualModule(14, dilation=1, dropout=0)
        )
        
        # MiniNetV5와 동일
        self.output_conv = nn.ConvTranspose2d(14, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # He 초기화 → Xavier 초기화 (GELU에 더 적합)
                nn.init.xavier_normal_(m.weight, gain=1.0)
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
        
        # 향상된 skip connections (파라미터 증가 없음)
        if up1.shape[2:] == skip.shape[2:]:
            up1 = 0.65 * up1 + 0.35 * skip  # 가중 결합
        if up1.shape[2:] == aux.shape[2:]:
            up1 = 0.85 * up1 + 0.15 * aux  # aux 비중 조정
            
        # Final processing
        m3 = self.upsample_mods(up1)
        out = self.output_conv(m3)
        
        # 향상된 interpolation
        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)  # align_corners 최적화
            
        return out

if __name__ == "__main__":
    # 테스트
    net = submission_MiniNetV7(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {params:,}")
    print(f"MiniNetV5 target: 7,009 params")
    print(f"Difference: {params - 7009:+,} params")
    print(f"Status: {'✅ PERFECT' if params == 7009 else '✅ CLOSE' if abs(params - 7009) <= 50 else '❌ ADJUST'}")
    
    # 개선사항 요약
    print("\n=== 파라미터 제로 증가 최적화 ===")
    print("1. ReLU → GELU (더 부드러운 활성화)")
    print("2. 가중 residual connection (성능 향상)")
    print("3. 가중 skip connection (정보 융합 최적화)")
    print("4. Xavier 초기화 (GELU에 최적)")
    print("5. BN momentum 조정 (더 안정적 훈련)")
    print("6. align_corners=False (더 정확한 보간)") 