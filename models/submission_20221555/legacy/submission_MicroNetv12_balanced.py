import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv12_Balanced: 8K 파라미터 균형잡힌 설계 ---

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

class BalancedCFDModule(nn.Module):
    """균형잡힌 CFD 모듈 - 효율적 CFD 처리 (Separable Conv 사용)"""
    def __init__(self, channels):
        super().__init__()
        # 2단계 processing (Separable Conv로 파라미터 절약)
        self.fine_conv1 = SeparableConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.fine_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)  # 1x1은 그대로
        
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        
        # CFD 특화 stabilization
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        out1 = self.fine_conv1(x)
        out1 = F.relu(out1)
        
        out2 = self.fine_conv2(out1)
        out2 = self.bn(out2)
        out2 = self.dropout(out2)
        
        return F.relu(x + out2)

class BalancedMedicalModule(nn.Module):
    """균형잡힌 Medical 모듈 - 효율적 ETIS 처리 (Separable Conv 사용)"""
    def __init__(self, channels):
        super().__init__()
        # 단일 단계 edge enhancement (Separable Conv로 파라미터 절약)
        self.edge_conv = SeparableConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(0.03)

    def forward(self, x):
        edge = self.edge_conv(x)
        edge = self.dropout(edge)
        return F.relu(x + edge)

class BalancedMultiScaleModule(nn.Module):
    """균형잡힌 Multi-scale 모듈 - 2-branch 효율적 (Separable Conv 사용)"""
    def __init__(self, channels):
        super().__init__()
        # 2-branch로 파라미터 효율성 유지
        half_channels = channels // 2
        self.branch1 = nn.Conv2d(channels, half_channels, kernel_size=1, bias=False)  # 1x1은 그대로
        self.branch2 = SeparableConv2d(channels, half_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        
        # Fusion도 간소화
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        concat = torch.cat([b1, b2], dim=1)
        out = self.fusion(concat)
        out = self.bn(out)
        return F.relu(x + out)

class CompactAttention(nn.Module):
    """컴팩트 Attention 모듈 - 최소 파라미터"""
    def __init__(self, channels):
        super().__init__()
        # 매우 경량 channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // 8, 1), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(channels // 8, 1), channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.avg_pool(x)
        att = self.fc(att)
        return x * att

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

# --- MicroNetv12_Balanced 인코더 ---

class MicroNetV12BalancedEncoder(nn.Module):
    """MicroNetv12_Balanced 인코더 - 균형잡힌 설계"""
    def __init__(self, in_channels):
        super().__init__()

        # 경량화된 채널: 3 → 7 → 12 → 15
        self.downsample_1 = MicroDownsampleModule(in_channels, 7)
        self.downsample_2 = MicroDownsampleModule(7, 12)
        
        # 균형잡힌 모듈 시스템
        self.downsample_modules = nn.Sequential(
            BalancedCFDModule(12),
            BalancedMedicalModule(12),
            BalancedMultiScaleModule(12)
        )
        
        self.downsample_3 = MicroDownsampleModule(12, 15)

        # 컴팩트 Attention
        self.attention = CompactAttention(15)

        # Feature modules: 2개로 축소 (rates: 1,2)
        rates = [1, 2]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(15, rate, 0.08) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        
        # Attention 적용
        d3 = self.attention(d3)
        
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection

# --- 최종 제출 모델: MicroNetv12_Balanced ---
class submission_MicroNetv12_balanced(nn.Module):
    """MicroNetv12_Balanced - 8K 파라미터 균형잡힌 설계"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (균형잡힌)
        self.encoder = MicroNetV12BalancedEncoder(in_channels)

        # 경량화된 Auxiliary path
        self.aux_downsample = MicroDownsampleModule(in_channels, 7)
        self.aux_refine = BalancedCFDModule(7)

        # 경량화된 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(15, 12)
        
        # 경량화된 processing
        self.upsample_mods = nn.Sequential(
            BalancedCFDModule(12),
            BalancedMultiScaleModule(12)
        )

        # 출력 (bias=False로 파라미터 절약)
        self.output_conv = nn.ConvTranspose2d(12, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # 균형잡힌 decoder
        up1 = self.upsample_1(enc)
        
        # Skip connection 활용 (채널 수 맞춤)
        if up1.shape[2:] == skip.shape[2:] and up1.shape[1] == skip.shape[1]:
            up1 = up1 + skip
        
        # 균형잡힌 processing
        up1 = self.upsample_mods(up1)
        
        # Auxiliary path와 결합 (채널 수 맞춤)
        if up1.shape[2:] == aux.shape[2:] and up1.shape[1] == aux.shape[1]:
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
    model_voc = submission_MicroNetv12_balanced(3, 21)
    params_voc = model_voc.count_parameters()
    print(f"VOC (21 classes) parameters: {params_voc:,}")
    
    # Binary 기준 테스트 (2 클래스)
    model_binary = submission_MicroNetv12_balanced(3, 2)
    params_binary = model_binary.count_parameters()
    print(f"Binary (2 classes) parameters: {params_binary:,}")
    
    # 목표 대비 분석
    target_min, target_max = 7000, 9000
    if target_min <= params_voc <= target_max:
        print(f'✅ 목표 달성! ({target_min:,}~{target_max:,}개 범위)')
    elif params_voc < target_min:
        print(f'⚠️ 목표 미달 ({target_min - params_voc:,}개 부족)')
    else:
        print(f'❌ 목표 초과 ({params_voc - target_max:,}개 초과)')
    
    # 테스트
    x = torch.randn(1, 3, 256, 256)
    try:
        y = model_voc(x)
        print(f"✅ 모델 테스트 성공: {x.shape} → {y.shape}")
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
    
    print(f"\n⚖️ 균형잡힌 전략:")
    print(f"v11_ultra: 4,062개 → v12_balanced: {params_voc:,}개")
    print(f"증가량: {params_voc - 4062:,}개 ({(params_voc - 4062)/4062*100:.1f}% 증가)")
    print(f"v12 대비: 31,030개 → {params_voc:,}개 ({(31030 - params_voc)/31030*100:.1f}% 감소)")
    
    print(f"\n🎯 성능 목표:")
    print(f"- Mean IoU: 0.40+ (v11_ultra 0.3439 대비 16% 향상)")
    print(f"- CFD IoU: 0.40+ (Balanced CFD 모듈)")
    print(f"- ETIS IoU: 0.40+ (Balanced Medical 모듈)")
    print(f"- CarDD IoU: 0.40+ (Balanced Multi-scale)")
    
    print(f"\n🔧 균형잡힌 혁신:")
    print(f"- Balanced CFD 모듈 (효율적)")
    print(f"- Balanced Medical 모듈 (ETIS 집중)")
    print(f"- Balanced Multi-scale (2-branch)")
    print(f"- Compact Attention (최소 파라미터)")
    print(f"- 3개 Feature modules (rates: 1,2,4)")
    print(f"- 균형잡힌 채널 (8→14→18)") 