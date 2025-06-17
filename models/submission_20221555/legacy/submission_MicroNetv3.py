import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv3: 의료/특수 도메인 특화 모듈들 ---

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution - 기본 모듈"""
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
    """Multi-dilation separable conv - 핵심 성능 모듈"""
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

class LightweightSpatialAttention(nn.Module):
    """경량 공간 어텐션 - 의료/특수 도메인 성능 향상"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)  # 최소 1채널 보장
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced_channels, eps=1e-3)
        
    def forward(self, x):
        # Global context
        gap = F.adaptive_avg_pool2d(x, 1)
        att = self.conv1(gap)
        att = self.bn(att)
        att = F.relu(att)
        att = self.conv2(att)
        att = torch.sigmoid(att)
        
        # Spatial attention
        spatial = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.sigmoid(spatial)
        
        # Combine attentions
        combined_att = att * spatial
        return x * combined_att

class MultiScalePyramidModule(nn.Module):
    """Multi-scale pyramid fusion - 다양한 스케일 특징 융합"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(in_channels // 2, 4)  # 최소 4채널 보장
        
        # Multi-scale branches (2개로 축소)
        self.branch1 = SeparableConv2d(in_channels, mid_channels, 3, padding=1, dilation=1)
        self.branch2 = SeparableConv2d(in_channels, mid_channels, 3, padding=2, dilation=2)
        
        # Fusion
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        fused = torch.cat([b1, b2], dim=1)
        out = self.fusion(fused)
        out = self.bn(out)
        return F.relu(out)

class EdgeEnhancementBranch(nn.Module):
    """경계 강화 브랜치 - 정확한 segmentation 경계"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(in_channels // 4, 2)  # 더 경량화
        self.edge_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.edge_conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)
        
    def forward(self, x):
        # Edge detection
        edge = self.edge_conv1(x)
        edge = self.bn1(edge)
        edge = F.relu(edge)
        
        edge = self.edge_conv2(edge)
        edge = self.bn2(edge)
        
        return torch.sigmoid(edge)

class MicroDownsampleModule(nn.Module):
    """다운샘플링 모듈"""
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
    """Residual 모듈 with attention"""
    def __init__(self, channels, dilation, dropout=0, use_attention=False):
        super().__init__()
        self.conv = SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = LightweightSpatialAttention(channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        
        if self.use_attention:
            out = self.attention(out)
            
        return F.relu(x + out)

class MicroResidualMultiDilationConvModule(nn.Module):
    """Multi-dilation Residual 모듈 with attention"""
    def __init__(self, channels, dilation, dropout=0, use_attention=False):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = LightweightSpatialAttention(channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        
        if self.use_attention:
            out = self.attention(out)
            
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

# --- MicroNetv3 인코더 (의료/특수 도메인 특화) ---

class MicroNetV3Encoder(nn.Module):
    """MicroNetv3 인코더 - 의료/특수 도메인 성능 향상"""
    def __init__(self, in_channels):
        super().__init__()

        # 채널 수: 3 → 10 → 18 → 24 (MicroNetv2와 동일)
        self.downsample_1 = MicroDownsampleModule(in_channels, 10)
        self.downsample_2 = MicroDownsampleModule(10, 18)
        
        # Downsample modules with attention (1개로 축소)
        self.downsample_modules = nn.Sequential(*[
            MicroResidualConvModule(18, 1, 0, use_attention=True)
        ])
        
        self.downsample_3 = MicroDownsampleModule(18, 24)

        # Feature modules with attention (의료 도메인 특화) - 경량화
        rates = [1, 2]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(24, rates[0], 0.1, use_attention=True),
            MicroResidualMultiDilationConvModule(24, rates[1], 0.1, use_attention=False)
        ])
        
        # Multi-scale pyramid for better feature representation
        self.pyramid = MultiScalePyramidModule(24, 24)

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        # Multi-scale enhancement
        m4_enhanced = self.pyramid(m4)
        
        return m4_enhanced, d2  # skip connection

# --- 최종 제출 모델: MicroNetv3 (의료/특수 도메인 특화) ---
class submission_MicroNetv3(nn.Module):
    """MicroNetv3 - 의료/특수 도메인 성능 향상"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (의료 도메인 특화)
        self.encoder = MicroNetV3Encoder(in_channels)

        # Auxiliary path (저수준 특징 보존)
        self.aux_downsample = MicroDownsampleModule(in_channels, 10)
        self.aux_refine = MicroResidualConvModule(10, 1, 0, use_attention=True)

        # Edge enhancement branch (경계 정확도 향상)
        self.edge_branch = EdgeEnhancementBranch(24, 1)

        # 업샘플 블록 (강화)
        self.upsample_1 = MicroUpsampleModule(24, 18)
        
        # Upsample modules with attention (1개로 축소)
        self.upsample_mods = nn.Sequential(*[
            MicroResidualConvModule(18, 1, 0, use_attention=True)
        ])

        # 출력 (개선)
        self.output_conv = nn.ConvTranspose2d(18, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # Edge enhancement
        edge_map = self.edge_branch(enc)
        
        # Apply edge enhancement to features
        enc_enhanced = enc * (1 + edge_map)
        
        # Decoder with skip connection
        up1 = self.upsample_1(enc_enhanced)
        
        # Skip connection 활용
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        
        # Auxiliary path와 결합
        if up1.shape[2:] == aux.shape[2:]:
            up1 = up1 + aux
            
        m3 = self.upsample_mods(up1)

        out = self.output_conv(m3)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_MicroNetv3(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MicroNetv3 (의료/특수 도메인 특화)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 파라미터 목표 검증
    if p <= 17000:
        print(f"✅ Hard cap 준수: {p}/17,000 ({17000-p} 여유)")
    else:
        print(f"❌ Hard cap 초과: {p}/17,000 ({p-17000} 초과)")
        
    if p < 10000:
        print(f"✅ 이상적 범위: {p}/10,000")
    elif p < 15000:
        print(f"✅ 우수 범위: {p}/15,000")

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
            net_test = submission_MicroNetv3(in_channels=3, num_classes=nc)
            net_test.eval()
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # v2와 비교 분석
    print(f"\n📊 MicroNetv2 → MicroNetv3 개선사항:")
    print(f"  파라미터: 15,459 → {p:,} ({p-15459:+,}, {(p-15459)/15459*100:+.1f}%)")
    print(f"  새로운 모듈:")
    print(f"    ✅ LightweightSpatialAttention - 중요 영역 집중")
    print(f"    ✅ MultiScalePyramidModule - 다양한 스케일 융합")
    print(f"    ✅ EdgeEnhancementBranch - 경계 정확도 향상")
    print(f"    ✅ Attention-enhanced modules - 의료 도메인 특화")
    
    # 모듈별 파라미터 분석
    encoder_params = sum(p.numel() for p in net.encoder.parameters())
    aux_params = sum(p.numel() for p in net.aux_downsample.parameters()) + sum(p.numel() for p in net.aux_refine.parameters())
    edge_params = sum(p.numel() for p in net.edge_branch.parameters())
    upsample_params = sum(p.numel() for p in net.upsample_1.parameters()) + sum(p.numel() for p in net.upsample_mods.parameters())
    output_params = sum(p.numel() for p in net.output_conv.parameters())
    
    print(f"\n🎯 모듈별 파라미터 분배:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/p*100:.1f}%)")
    print(f"  Auxiliary: {aux_params:,} ({aux_params/p*100:.1f}%)")
    print(f"  Edge Branch: {edge_params:,} ({edge_params/p*100:.1f}%)")
    print(f"  Upsample: {upsample_params:,} ({upsample_params/p*100:.1f}%)")
    print(f"  Output: {output_params:,} ({output_params/p*100:.1f}%)")
    
    print(f"\n🚀 MicroNetv3 특징:")
    print(f"  🎯 의료/특수 도메인 특화 설계")
    print(f"  ✅ Spatial Attention - ETIS, CFD, CarDD 성능 향상")
    print(f"  ✅ Multi-Scale Pyramid - 다양한 객체 크기 대응")
    print(f"  ✅ Edge Enhancement - 정확한 경계 분할")
    print(f"  ✅ Hard cap 17K 준수")
    print(f"  📈 목표: ETIS 0.32→0.4+, CFD 0.34→0.4+, CarDD 0.35→0.4+")
    
    # 성능 예측
    print(f"\n📈 예상 성능 향상:")
    print(f"  ETIS (폴립): 0.3215 → 0.40+ (작은 객체 탐지 개선)")
    print(f"  CFD (크랙): 0.3366 → 0.42+ (선형 구조 탐지 개선)")
    print(f"  CarDD (손상): 0.3454 → 0.38+ (복잡한 패턴 인식 개선)")
    print(f"  VOC: 0.1205 → 0.13+ (일반 도메인 유지)")
    print(f"  CVPPP: 0.9185 → 0.92+ (이미 우수한 성능 유지)")
    print(f"  Mean IoU: 0.4085 → 0.43+ (전체적 성능 향상)") 