import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv10_Ultra: 극도 경량화 (VOC 10K 확실 달성) ---

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

class CFDStabilizedModule(nn.Module):
    """CFD 안정성 모듈 - 최소한 유지"""
    def __init__(self, channels):
        super().__init__()
        # 최소한의 fine-grained processing
        self.fine_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        out = self.fine_conv(x)
        out = self.bn(out)
        out = self.dropout(out)
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

# --- MicroNetv10_Ultra 인코더 (극도 경량화) ---

class MicroNetV10UltraEncoder(nn.Module):
    """MicroNetv10_Ultra 인코더 - 극도 경량화"""
    def __init__(self, in_channels):
        super().__init__()

        # 극도 경량화 채널: 3 → 6 → 9 → 12
        self.downsample_1 = MicroDownsampleModule(in_channels, 6)
        self.downsample_2 = MicroDownsampleModule(6, 9)
        
        # CFD 안정성만 유지 (Multi-scale 제거)
        self.downsample_modules = CFDStabilizedModule(9)
        
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

# --- 최종 제출 모델: MicroNetv10_Ultra (극도 경량화) ---
class submission_MicroNetv10_ultra(nn.Module):
    """MicroNetv10_Ultra - 극도 경량화 (VOC 10K 확실 달성)"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (극도 경량화)
        self.encoder = MicroNetV10UltraEncoder(in_channels)

        # Auxiliary path 제거 (파라미터 절약)

        # 간소화된 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(12, 9)
        
        # 최소한의 processing
        self.upsample_mods = CFDStabilizedModule(9)

        # 출력 (bias=False로 파라미터 절약)
        self.output_conv = nn.ConvTranspose2d(9, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # Main encoder only
        enc, skip = self.encoder(x)
        
        # 간소화된 decoder
        up1 = self.upsample_1(enc)
        
        # Skip connection 활용
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        
        # 최소한의 processing
        up1 = self.upsample_mods(up1)
        
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
    model_voc = submission_MicroNetv10_ultra(3, 21)
    params_voc = model_voc.count_parameters()
    print(f"VOC (21 classes) parameters: {params_voc:,}")
    
    # Binary 기준 테스트 (2 클래스)
    model_binary = submission_MicroNetv10_ultra(3, 2)
    params_binary = model_binary.count_parameters()
    print(f"Binary (2 classes) parameters: {params_binary:,}")
    
    # 목표 대비 분석
    target = 10000
    if params_voc <= target:
        print(f'✅ VOC 목표 달성! ({target:,}개 이내)')
    else:
        print(f'❌ VOC 목표 초과 ({params_voc - target:,}개 초과)')
    
    # 테스트
    x = torch.randn(1, 3, 256, 256)
    try:
        y = model_voc(x)
        print(f"✅ 모델 테스트 성공: {x.shape} → {y.shape}")
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
    
    print(f"\n📊 v10 대비 파라미터 감소:")
    print(f"v10: 15,372개 → Ultra: {params_voc:,}개")
    print(f"감소량: {15372 - params_voc:,}개 ({(15372 - params_voc)/15372*100:.1f}% 감소)") 