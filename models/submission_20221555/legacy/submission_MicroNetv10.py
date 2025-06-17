import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv10: Balanced (VOC 10K 이하 + 성능 회복) ---

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

class EnhancedCFDStabilizedModule(nn.Module):
    """강화된 CFD 안정성 모듈 - 성능 개선 집중"""
    def __init__(self, channels):
        super().__init__()
        # Enhanced fine-grained processing
        self.fine_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.fine_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)
        
        # Gradient stabilization
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        # Multi-step fine processing
        out1 = self.fine_conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1)
        
        out2 = self.fine_conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.dropout(out2)
        
        return F.relu(x + out2)

class BalancedMultiScaleModule(nn.Module):
    """균형잡힌 Multi-scale 모듈 - 3-branch로 성능 회복"""
    def __init__(self, channels):
        super().__init__()
        # 3개 branch로 표현력 향상
        branch_channels = channels // 3
        self.branch1 = nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False)
        self.branch2 = SeparableConv2d(channels, branch_channels, kernel_size=3, padding=1, bias=False)
        self.branch3 = SeparableConv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        concat = torch.cat([b1, b2, b3], dim=1)
        out = self.fusion(concat)
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

# --- MicroNetv10 인코더 (Balanced) ---

class MicroNetV10Encoder(nn.Module):
    """MicroNetv10 인코더 - Balanced (성능 회복 집중)"""
    def __init__(self, in_channels):
        super().__init__()

        # 균형잡힌 채널: 3 → 8 → 14 → 18
        self.downsample_1 = MicroDownsampleModule(in_channels, 8)
        self.downsample_2 = MicroDownsampleModule(8, 14)
        
        # 강화된 모듈들
        self.downsample_modules = nn.Sequential(
            EnhancedCFDStabilizedModule(14),
            BalancedMultiScaleModule(14)
        )
        
        self.downsample_3 = MicroDownsampleModule(14, 18)

        # Feature modules 복원: 4개 (rates: 1,2,4,6)
        rates = [1, 2, 4, 6]
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

# --- 최종 제출 모델: MicroNetv10 (Balanced) ---
class submission_MicroNetv10(nn.Module):
    """MicroNetv10 - Balanced (VOC 10K 이하 + 성능 회복)"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (Balanced)
        self.encoder = MicroNetV10Encoder(in_channels)

        # 간소화된 Auxiliary path (파라미터 절약)
        self.aux_downsample = MicroDownsampleModule(in_channels, 8)
        self.aux_refine = EnhancedCFDStabilizedModule(8)

        # 균형잡힌 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(18, 14)
        
        # 강화된 upsample processing
        self.upsample_mods = nn.Sequential(
            EnhancedCFDStabilizedModule(14),
            BalancedMultiScaleModule(14)
        )

        # 출력 (bias=False로 파라미터 절약)
        self.output_conv = nn.ConvTranspose2d(14, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # 균형잡힌 decoder
        up1 = self.upsample_1(enc)
        
        # Skip connection 활용
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        
        # 강화된 processing
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
    model_voc = submission_MicroNetv10(3, 21)
    params_voc = model_voc.count_parameters()
    print(f"VOC (21 classes) parameters: {params_voc:,}")
    
    # Binary 기준 테스트 (2 클래스)
    model_binary = submission_MicroNetv10(3, 2)
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
    y = model_voc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}") 