import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv7_lite: 극도 경량화 + CFD 안정성 ---

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

class LiteCFDStabilizedModule(nn.Module):
    """경량화된 CFD 안정성 모듈"""
    def __init__(self, channels):
        super().__init__()
        # 매우 간단한 fine-grained processing
        self.fine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3)
        )
        
        # 안정적인 gradient flow
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        out = self.fine_conv(x)
        out = self.dropout(out)
        return F.relu(x + out)

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

# --- MicroNetv7_lite 인코더 (극도 경량화) ---

class MicroNetV7LiteEncoder(nn.Module):
    """MicroNetv7_lite 인코더 - 극도 경량화 + CFD 안정성"""
    def __init__(self, in_channels):
        super().__init__()

        # 채널 수 대폭 감소: 3 → 8 → 14 → 18 (v7: 10→18→24에서 감소)
        self.downsample_1 = MicroDownsampleModule(in_channels, 8)
        self.downsample_2 = MicroDownsampleModule(8, 14)
        
        # Downsample modules: 1개로 감소 + CFD 모듈
        self.downsample_modules = nn.Sequential(*[
            LiteCFDStabilizedModule(14)  # 경량화된 CFD 안정성 모듈
        ])
        
        self.downsample_3 = MicroDownsampleModule(14, 18)

        # Feature modules: 3개로 감소 (v7: 4개에서 감소)
        rates = [1, 2, 4]  # 8 제거
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(18, rate, 0.08) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection

# --- 최종 제출 모델: MicroNetv7_lite (극도 경량화) ---
class submission_MicroNetv7_lite(nn.Module):
    """MicroNetv7_lite - 극도 경량화 + CFD 안정성"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (극도 경량화)
        self.encoder = MicroNetV7LiteEncoder(in_channels)

        # 간소화된 auxiliary path
        self.aux_downsample = MicroDownsampleModule(in_channels, 8)
        self.aux_refine = LiteCFDStabilizedModule(8)  # 경량화된 CFD 안정성

        # 간단한 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(18, 14)
        
        # 최소한의 upsample modules
        self.upsample_mods = LiteCFDStabilizedModule(14)  # 1개로 감소

        # 출력 (간소화)
        self.output_conv = nn.ConvTranspose2d(14, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # 간단한 decoder
        up1 = self.upsample_1(enc)
        
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
    net = submission_MicroNetv7_lite(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MicroNetv7_lite (극도 경량화 + CFD 안정성)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 파라미터 목표 검증 (Hard cap: 17K)
    if p <= 15000:
        print(f"✅ 이상적 범위: {p}/15,000")
    elif p <= 17000:
        print(f"✅ Hard cap 내: {p}/17,000 ({17000-p} 여유)")
    elif p <= 20000:
        print(f"⚠️  목표 초과: {p}/17,000 ({p-17000} 초과)")
    else:
        print(f"❌ 크게 초과: {p}/17,000 ({p-17000} 초과)")

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
            net_test = submission_MicroNetv7_lite(in_channels=3, num_classes=nc)
            net_test.eval()
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # 이전 버전들과 비교 분석
    print(f"\n📊 모델 진화 분석:")
    print(f"  MicroNetv5: 15,461개 → 0.4247 IoU (성공)")
    print(f"  MicroNetv6: 22,595개 → 0.3819 IoU (복잡성 과다)")
    print(f"  MicroNetv7: 24,945개 → 목표 실패 (파라미터 초과)")
    print(f"  MicroNetv7_lite: {p:,}개 → 목표: 0.40+ IoU (극도 경량화)")
    
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
    
    print(f"\n🚀 MicroNetv7_lite 극도 경량화:")
    print(f"  ✅ 채널 수 대폭 감소: 8→14→18 (v7: 10→18→24)")
    print(f"  ✅ Feature modules 감소: 3개 (v7: 4개)")
    print(f"  ✅ LiteCFDStabilizedModule - 경량화된 CFD 안정성")
    print(f"  ✅ Multi-scale 모듈 제거 - 파라미터 절약")
    print(f"  ✅ Upsample modules 최소화")
    
    print(f"\n📈 예상 성능:")
    print(f"  CFD 안정성: 경량화된 모듈로도 안정적 학습")
    print(f"  전체 성능: 0.40+ IoU (v5 수준 목표)")
    print(f"  파라미터 효율성: 극도로 효율적")
    print(f"  학습 안정성: CFD 초기 학습 개선")
    
    print(f"\n🔍 극도 경량화 전략:")
    print(f"  🎯 채널 수 최소화: 필수 표현력만 유지")
    print(f"  🎯 모듈 수 최소화: 핵심 기능만 보존")
    print(f"  🎯 CFD 안정성 유지: 경량화해도 핵심 기능 보존")
    print(f"  🎯 Skip connection 유지: 정보 손실 최소화")
    
    print(f"\n✨ MicroNetv7_lite 설계 철학:")
    print(f"  🎯 극도 경량화: 17K 이하 반드시 달성")
    print(f"  🔧 CFD 안정성: 핵심 기능은 반드시 유지")
    print(f"  ⚡ 최소 구조: 불필요한 모든 것 제거")
    print(f"  🚀 실용성 극대화: 파라미터 대비 최대 성능")
    
    # 경량화 비교
    print(f"\n📊 경량화 비교:")
    print(f"  v5 → v7_lite: 15,461 → {p:,} ({p-15461:+,})")
    print(f"  v6 → v7_lite: 22,595 → {p:,} ({(22595-p)/22595*100:.1f}% 감소)")
    print(f"  v7 → v7_lite: 24,945 → {p:,} ({(24945-p)/24945*100:.1f}% 감소)")
    
    print(f"\n🎯 성능 예측 (보수적):")
    print(f"  VOC: 0.10+ (경량화로 약간 하락 예상)")
    print(f"  ETIS: 0.35+ (안정적 유지)")
    print(f"  CVPPP: 0.90+ (높은 수준 유지)")
    print(f"  CFD: 0.20+ (안정성 모듈로 개선)")
    print(f"  CarDD: 0.35+ (안정적 유지)")
    print(f"  전체: 0.40+ IoU (v5 수준 목표)")
    
    print(f"\n🔧 CFD 안정성 유지 전략:")
    print(f"  LiteCFDStabilizedModule: 경량하지만 효과적")
    print(f"  Fine-grained processing: 작은 패턴 학습")
    print(f"  안정적 gradient flow: residual + dropout")
    print(f"  초기 학습 개선: 0.0000 구간 탈출") 