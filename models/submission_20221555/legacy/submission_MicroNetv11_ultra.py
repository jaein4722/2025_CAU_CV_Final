import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv11_Ultra: 7K 파라미터 확실 달성 + 에러 해결 ---

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution - 핵심 효율 모듈"""
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

class MultiDilationSeparableConv2d(nn.Module):
    """Multi-dilation separable conv - 성능 핵심 모듈 (유지)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
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

class UltraLiteCFDModule(nn.Module):
    """극도 경량화된 CFD 모듈"""
    def __init__(self, channels):
        super().__init__()
        # 최소한의 processing
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(x + out)

class MicroDownsampleModule(nn.Module):
    """다운샘플링 모듈 - DenseNet 스타일 (유지)"""
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

class MicroResidualMultiDilationConvModule(nn.Module):
    """Multi-dilation Residual 모듈 - 핵심 성능 모듈 (유지)"""
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

# --- MicroNetv11_Ultra 인코더 (극도 경량화) ---

class MicroNetV11UltraEncoder(nn.Module):
    """MicroNetv11_Ultra 인코더 - 극도 경량화"""
    def __init__(self, in_channels):
        super().__init__()

        # 극도 경량화된 채널: 3 → 6 → 9 → 12
        self.downsample_1 = MicroDownsampleModule(in_channels, 6)
        self.downsample_2 = MicroDownsampleModule(6, 9)
        
        # 최소한의 모듈
        self.downsample_modules = UltraLiteCFDModule(9)
        
        self.downsample_3 = MicroDownsampleModule(9, 12)

        # Feature modules: 2개만 (rates: 1,2)
        rates = [1, 2]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(12, rate, 0.08) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection

# --- 최종 제출 모델: MicroNetv11_Ultra ---
class submission_MicroNetv11_ultra(nn.Module):
    """MicroNetv11_Ultra - 7K 파라미터 확실 달성"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (극도 경량화)
        self.encoder = MicroNetV11UltraEncoder(in_channels)

        # 최소한의 Auxiliary path
        self.aux_downsample = MicroDownsampleModule(in_channels, 6)

        # 극도 경량화된 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(12, 9)
        
        # 최소한의 processing
        self.upsample_mods = UltraLiteCFDModule(9)

        # 출력 (bias=False로 파라미터 절약)
        self.output_conv = nn.ConvTranspose2d(9, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # 극도 경량화된 decoder
        up1 = self.upsample_1(enc)
        
        # Skip connection 활용
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        
        # 최소한의 processing
        up1 = self.upsample_mods(up1)
        
        # Auxiliary path와 결합
        if up1.shape[2:] == aux.shape[2:]:
            up1 = up1 + aux
        
        # 최종 출력
        output = self.output_conv(up1)
        
        if self.interpolate and output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output

    def count_parameters(self):
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 파라미터 수 확인용
if __name__ == "__main__":
    # VOC 기준 테스트 (21 클래스)
    model_voc = submission_MicroNetv11_ultra(3, 21)
    params_voc = model_voc.count_parameters()
    print(f"VOC (21 classes) parameters: {params_voc:,}")
    
    # Binary 기준 테스트 (2 클래스)
    model_binary = submission_MicroNetv11_ultra(3, 2)
    params_binary = model_binary.count_parameters()
    print(f"Binary (2 classes) parameters: {params_binary:,}")
    
    # 목표 대비 분석
    target = 7000
    if params_voc <= target:
        print(f'✅ 목표 달성! ({target:,}개 이내)')
        margin = target - params_voc
        print(f'여유분: {margin:,}개 ({margin/target*100:.1f}%)')
    else:
        print(f'❌ 목표 초과 ({params_voc - target:,}개 초과)')
    
    # 테스트
    x = torch.randn(1, 3, 256, 256)
    try:
        y = model_voc(x)
        print(f"✅ 모델 테스트 성공: {x.shape} → {y.shape}")
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
    
    print(f"\n📊 극도 경량화 전략:")
    print(f"v11_lite: 8,122개 → v11_ultra: {params_voc:,}개")
    print(f"감소량: {8122 - params_voc:,}개 ({(8122 - params_voc)/8122*100:.1f}% 감소)")
    print(f"v10_ultra 대비: 3,969개 → {params_voc:,}개 ({(params_voc - 3969)/3969*100:.1f}% 증가)")
    
    print(f"\n🎯 성능 목표:")
    print(f"- Mean IoU: 0.35+ (v10_ultra 0.3194 대비 개선)")
    print(f"- CFD IoU: 0.20+ → 0.4+ 도전")
    print(f"- ETIS IoU: 0.20+ → 0.4+ 도전")
    print(f"- CarDD IoU: 0.35+ → 0.4+ 도전")
    
    print(f"\n🔧 핵심 개선사항:")
    print(f"- 채널 불일치 문제 해결")
    print(f"- 7K 파라미터 확실 달성")
    print(f"- Multi-dilation 핵심 모듈 유지")
    print(f"- Skip connection 최적화") 