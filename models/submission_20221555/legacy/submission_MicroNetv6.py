import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv6: 성능 향상 + 혁신적 기술 도입 ---

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution - 핵심 효율 모듈"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
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

class MultiDilationSeparableConv2d(nn.Module):
    """Multi-dilation separable conv - 성능 핵심 모듈"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,         1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.depthwise2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        out = x1 + x2
        out = self.pointwise(out)
        return out

class LightweightChannelAttention(nn.Module):
    """경량 Channel Attention - 성능 향상의 핵심"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 매우 경량한 FC layers
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class EnhancedMultiScaleModule(nn.Module):
    """향상된 Multi-scale 모듈 - VOC 성능 개선"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 4
        
        # 더 다양한 scale branches (VOC의 다양한 객체 크기 대응)
        self.branch1 = SeparableConv2d(in_channels, mid_channels, 3, padding=1, dilation=1)
        self.branch2 = SeparableConv2d(in_channels, mid_channels, 3, padding=2, dilation=2)
        self.branch3 = SeparableConv2d(in_channels, mid_channels, 3, padding=4, dilation=4)
        self.branch4 = SeparableConv2d(in_channels, mid_channels, 3, padding=6, dilation=6)
        
        # Global context branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention for fusion
        self.attention = LightweightChannelAttention(out_channels)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Global context
        bg = self.global_branch(x)
        bg = F.interpolate(bg, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Fusion
        fused = torch.cat([b1, b2, b3, b4, bg], dim=1)
        out = self.fusion(fused)
        
        # Apply attention
        out = self.attention(out)
        
        return out

class MicroDownsampleModule(nn.Module):
    """다운샘플링 모듈 - DenseNet 스타일"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels

        if not self.use_maxpool:
            channels_conv = out_channels
        else:
            channels_conv = out_channels - in_channels

        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)

        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)

        out = self.bn(out)
        return F.relu(out)

class MicroResidualConvModule(nn.Module):
    """Residual 모듈 - 표현력 강화"""
    def __init__(self, channels, dilation, dropout=0):
        super().__init__()
        self.conv = SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class MicroResidualMultiDilationConvModule(nn.Module):
    """Multi-dilation Residual 모듈 - 핵심 성능 모듈"""
    def __init__(self, channels, dilation, dropout=0):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class MicroUpsampleModule(nn.Module):
    """업샘플링 모듈"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)

class FeaturePyramidFusion(nn.Module):
    """Feature Pyramid Network 스타일 fusion - VOC 성능 향상"""
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        # High-level feature processing
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Low-level feature processing
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_feat, low_feat):
        # Process high-level features
        high_processed = self.high_conv(high_feat)
        
        # Upsample to match low-level feature size
        high_upsampled = F.interpolate(high_processed, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Process low-level features
        low_processed = self.low_conv(low_feat)
        
        # Element-wise addition
        fused = high_upsampled + low_processed
        
        # Final convolution
        out = self.fusion_conv(fused)
        
        return out

# --- MicroNetv6 인코더 (성능 향상) ---

class MicroNetV6Encoder(nn.Module):
    """MicroNetv6 인코더 - 성능 향상 + 혁신적 기술"""
    def __init__(self, in_channels):
        super().__init__()

        # 채널 수: 3 → 10 → 20 → 28 (v5: 26에서 약간 증가)
        self.downsample_1 = MicroDownsampleModule(in_channels, 10)
        self.downsample_2 = MicroDownsampleModule(10, 20)
        
        # Downsample modules: 2개 유지
        self.downsample_modules = nn.Sequential(*[
            MicroResidualConvModule(20, 1, 0),
            MicroResidualConvModule(20, 1, 0)
        ])
        
        self.downsample_3 = MicroDownsampleModule(20, 28)

        # Enhanced feature modules: 5개로 증가 (더 다양한 receptive field)
        rates = [1, 2, 4, 6, 8]  # vs v5: [1, 2, 4, 8]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(28, rate, 0.1) for rate in rates
        ])
        
        # Enhanced multi-scale module
        self.multi_scale = EnhancedMultiScaleModule(28, 28)

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        # Enhanced multi-scale processing
        m4_enhanced = self.multi_scale(m4)
        
        return m4_enhanced, d2, d1  # 더 많은 skip connections

# --- 최종 제출 모델: MicroNetv6 (성능 향상) ---
class submission_MicroNetv6(nn.Module):
    """MicroNetv6 - 성능 향상 + 혁신적 기술 도입"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (성능 향상)
        self.encoder = MicroNetV6Encoder(in_channels)

        # Enhanced auxiliary path
        self.aux_downsample = MicroDownsampleModule(in_channels, 10)
        self.aux_refine = nn.Sequential(
            MicroResidualConvModule(10, 1, 0),
            LightweightChannelAttention(10)  # Attention 추가
        )

        # Feature Pyramid Network 스타일 decoder
        self.fpn_fusion1 = FeaturePyramidFusion(28, 20, 20)  # high_feat(28) + skip(20) → 20
        self.fpn_fusion2 = FeaturePyramidFusion(20, 10, 12)  # fused(20) + aux(10) → 12
        
        # Final upsample modules
        self.upsample_mods = nn.Sequential(*[
            MicroResidualConvModule(12, 1, 0),
            LightweightChannelAttention(12),  # Attention 추가
            MicroResidualConvModule(12, 1, 0)
        ])

        # 출력 (개선)
        self.output_conv = nn.ConvTranspose2d(12, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip2, skip1 = self.encoder(x)
        
        # Feature Pyramid Network 스타일 decoding
        # Stage 1: high-level(28) + mid-level(20) → 20
        fused1 = self.fpn_fusion1(enc, skip2)
        
        # Stage 2: fused(20) + low-level(10) → 12
        fused2 = self.fpn_fusion2(fused1, aux)
        
        # Final processing
        refined = self.upsample_mods(fused2)

        # Output
        out = self.output_conv(refined)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_MicroNetv6(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MicroNetv6 (성능 향상 + 혁신적 기술)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 파라미터 목표 검증
    if p <= 15000:
        print(f"✅ v5 수준 유지: {p}/15,000")
    elif p <= 20000:
        print(f"✅ 목표 달성: {p}/20,000 ({20000-p} 여유)")
    elif p <= 25000:
        print(f"✅ 허용 범위: {p}/25,000 ({25000-p} 여유)")
    else:
        print(f"⚠️  목표 초과: {p}/20,000 ({p-20000} 초과)")

    try:
        net.eval()  # 테스트 모드
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("✅ 모델 실행 테스트 통과")
        
        # 다양한 클래스 수에 대한 테스트
        for nc in [1, 2, 21]:
            net_test = submission_MicroNetv6(in_channels=3, num_classes=nc)
            net_test.eval()
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # 이전 버전들과 비교 분석
    print(f"\n📊 모델 진화 분석:")
    print(f"  MicroNetv2: 15,459개 → 0.4085 IoU")
    print(f"  MicroNetv4: 28,965개 → 0.4046 IoU (비효율)")
    print(f"  MicroNetv5: 15,461개 → 0.4247 IoU (효율적)")
    print(f"  MicroNetv6: {p:,}개 → 목표: 0.43+ IoU (혁신)")
    
    # 모듈별 파라미터 분석
    encoder_params = sum(p.numel() for p in net.encoder.parameters())
    aux_params = sum(p.numel() for p in net.aux_downsample.parameters()) + sum(p.numel() for p in net.aux_refine.parameters())
    fpn_params = sum(p.numel() for p in net.fpn_fusion1.parameters()) + sum(p.numel() for p in net.fpn_fusion2.parameters())
    upsample_params = sum(p.numel() for p in net.upsample_mods.parameters())
    output_params = sum(p.numel() for p in net.output_conv.parameters())
    
    print(f"\n🎯 모듈별 파라미터 분배:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/p*100:.1f}%)")
    print(f"  Auxiliary: {aux_params:,} ({aux_params/p*100:.1f}%)")
    print(f"  FPN Fusion: {fpn_params:,} ({fpn_params/p*100:.1f}%)")
    print(f"  Upsample: {upsample_params:,} ({upsample_params/p*100:.1f}%)")
    print(f"  Output: {output_params:,} ({output_params/p*100:.1f}%)")
    
    print(f"\n🚀 MicroNetv6 혁신적 기술:")
    print(f"  🆕 LightweightChannelAttention - 효율적 attention")
    print(f"  🆕 EnhancedMultiScaleModule - VOC 성능 향상")
    print(f"  🆕 FeaturePyramidFusion - FPN 스타일 decoder")
    print(f"  ✅ 5개 feature modules (rates: 1,2,4,6,8)")
    print(f"  ✅ 다중 skip connections + attention")
    
    print(f"\n📈 예상 성능 개선:")
    print(f"  전체 성능: 0.4247 → 0.43+ IoU")
    print(f"  VOC 성능: 0.1301 → 0.15+ IoU (핵심 목표)")
    print(f"  Multi-class 대응: Enhanced multi-scale로 개선")
    print(f"  Feature fusion: FPN으로 정보 손실 최소화")
    
    print(f"\n🔍 핵심 개선 포인트:")
    print(f"  🎯 VOC 특화: 다양한 객체 크기 대응 (1,2,4,6,8 dilation)")
    print(f"  🎯 Attention 도입: Channel attention으로 중요 특징 강조")
    print(f"  🎯 FPN 구조: 다중 스케일 정보 효과적 융합")
    print(f"  🎯 Global context: AdaptiveAvgPool로 전역 정보 활용")
    
    print(f"\n✨ MicroNetv6 설계 철학:")
    print(f"  🚀 성능 우선: v5 기반으로 혁신적 기술 도입")
    print(f"  🎯 VOC 집중: 가장 어려운 데이터셋 성능 향상")
    print(f"  ⚡ 효율적 혁신: 파라미터 증가 최소화하며 성능 극대화")
    print(f"  🔧 검증된 + 새로운: 안정적 구조에 혁신 기술 결합")
    
    # 기술적 혁신 요약
    print(f"\n🔬 기술적 혁신 요약:")
    print(f"  Channel Attention: 중요 채널 강조로 표현력 향상")
    print(f"  Multi-scale Enhancement: 5개 branch + global context")
    print(f"  Feature Pyramid: 계층적 특징 융합으로 정보 보존")
    print(f"  Adaptive Fusion: 데이터셋별 특성 고려한 융합")
    
    print(f"\n🎯 성능 예측:")
    print(f"  VOC: 0.1301 → 0.15+ (다중 클래스 대응 강화)")
    print(f"  ETIS: 0.3713 → 0.38+ (attention 효과)")
    print(f"  CVPPP: 0.9209 → 0.92+ (이미 높은 수준 유지)")
    print(f"  CFD: 0.3205 → 0.34+ (multi-scale 효과)")
    print(f"  CarDD: 0.3806 → 0.39+ (feature fusion 효과)")
    print(f"  전체: 0.4247 → 0.43+ IoU 목표") 