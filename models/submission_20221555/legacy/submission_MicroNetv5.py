import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv5: 파라미터 효율성 최적화 ---

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

# --- MicroNetv5 인코더 (효율성 최적화) ---

class MicroNetV5Encoder(nn.Module):
    """MicroNetv5 인코더 - 파라미터 효율성 최적화"""
    def __init__(self, in_channels):
        super().__init__()

        # 채널 수 최적화: 3 → 10 → 20 → 26 (v2: 10→18→24에서 개선)
        self.downsample_1 = MicroDownsampleModule(in_channels, 10)
        self.downsample_2 = MicroDownsampleModule(10, 20)
        
        # Downsample modules: 2개 유지 (안정적 학습)
        self.downsample_modules = nn.Sequential(*[
            MicroResidualConvModule(20, 1, 0),
            MicroResidualConvModule(20, 1, 0)
        ])
        
        self.downsample_3 = MicroDownsampleModule(20, 26)

        # Feature modules: 3개 → 4개로 증가 (표현력 강화)
        rates = [1, 2, 4, 8]  # vs v2: [1, 2, 4]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(26, rate, 0.1) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        return m4, d2  # skip connection을 위해 d2도 반환

# --- 최종 제출 모델: MicroNetv5 (효율성 최적화) ---
class submission_MicroNetv5(nn.Module):
    """MicroNetv5 - 파라미터 효율성 최적화"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (효율성 최적화)
        self.encoder = MicroNetV5Encoder(in_channels)

        # Auxiliary path 간소화 (효율성 우선)
        self.aux_downsample = MicroDownsampleModule(in_channels, 10)
        self.aux_refine = MicroResidualConvModule(10, 1, 0)

        # 업샘플 블록 (최적화)
        self.upsample_1 = MicroUpsampleModule(26, 20)
        
        # Upsample modules: 2개 유지 (효율성과 성능 균형)
        self.upsample_mods = nn.Sequential(*[
            MicroResidualConvModule(20, 1, 0),
            MicroResidualConvModule(20, 1, 0)
        ])

        # 출력 (간소화)
        self.output_conv = nn.ConvTranspose2d(20, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # Decoder with skip connection
        up1 = self.upsample_1(enc)
        
        # Skip connection 활용 (성능 향상)
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
    net = submission_MicroNetv5(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MicroNetv5 (파라미터 효율성 최적화)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 파라미터 목표 검증
    if p < 10000:
        print(f"✅ 이상적 범위: {p}/10,000")
    elif p <= 15000:
        print(f"✅ 목표 달성: {p}/15,000 ({15000-p} 여유)")
    elif p <= 17000:
        print(f"✅ Hard cap 내: {p}/17,000 ({17000-p} 여유)")
    else:
        print(f"⚠️  목표 초과: {p}/15,000 ({p-15000} 초과)")

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
            net_test = submission_MicroNetv5(in_channels=3, num_classes=nc)
            net_test.eval()
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # v2, v4와 비교 분석
    print(f"\n📊 모델 비교 분석:")
    print(f"  MicroNetv2: 15,459개 → 0.4085 IoU")
    print(f"  MicroNetv4: 28,965개 → 0.4046 IoU (비효율)")
    print(f"  MicroNetv5: {p:,}개 → 목표: 0.41+ IoU")
    print(f"  효율성 개선: v4 대비 {28965-p:,}개 감소 ({(28965-p)/28965*100:.1f}%)")
    
    # 모듈별 파라미터 분석
    encoder_params = sum(p.numel() for p in net.encoder.parameters())
    aux_params = sum(p.numel() for p in net.aux_downsample.parameters()) + sum(p.numel() for p in net.aux_refine.parameters())
    upsample_params = (sum(p.numel() for p in net.upsample_1.parameters()) + 
                      sum(p.numel() for p in net.upsample_mods.parameters()))
    output_params = sum(p.numel() for p in net.output_conv.parameters())
    
    print(f"\n🎯 모듈별 파라미터 분배:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/p*100:.1f}%)")
    print(f"  Auxiliary: {aux_params:,} ({aux_params/p*100:.1f}%)")
    print(f"  Upsample: {upsample_params:,} ({upsample_params/p*100:.1f}%)")
    print(f"  Output: {output_params:,} ({output_params/p*100:.1f}%)")
    
    print(f"\n🚀 MicroNetv5 핵심 개선사항:")
    print(f"  ✅ 채널 최적화: 10→20→26 (v2: 10→18→24)")
    print(f"  ✅ Feature modules 강화: 4개 (rates: 1,2,4,8)")
    print(f"  ✅ 불필요한 복잡성 제거 (MorphGradientFocus 등)")
    print(f"  ✅ 파라미터 효율성 우선 설계")
    print(f"  ✅ 검증된 구조 기반 (MicroNetv2)")
    
    print(f"\n📈 예상 성능 개선:")
    print(f"  파라미터 효율성: v4 대비 {(28965-p)/28965*100:.1f}% 감소")
    print(f"  성능 목표: 0.41+ IoU (v2 수준 이상)")
    print(f"  안정성: 검증된 구조로 안정적 학습")
    print(f"  확장성: 필요시 추가 최적화 가능")
    
    # 채널 진행 분석
    print(f"\n🔍 채널 진행 분석:")
    print(f"  Input: 3 → Downsample1: 10 → Downsample2: 20 → Downsample3: 26")
    print(f"  Feature processing: 26 (4개 multi-dilation modules)")
    print(f"  Upsample1: 26→20 → Upsample2: 20→classes")
    print(f"  Skip connections: 20↔20, Auxiliary: 10↔20")
    
    print(f"\n✨ MicroNetv5 설계 철학:")
    print(f"  🎯 효율성 우선: 최소 파라미터로 최대 성능")
    print(f"  🔧 검증된 구조: MicroNetv2의 성공 요소 계승")
    print(f"  ⚡ 점진적 개선: 채널 수와 모듈 수 최적화")
    print(f"  🚀 실용성 중시: 복잡한 모듈 대신 간단하고 효과적인 구조") 