import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv9: Ultra-Lite (10K 파라미터 목표) ---

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
    """Multi-dilation separable conv - 성능 핵심 모듈 (유지)"""
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

class CFDStabilizedModule(nn.Module):
    """CFD 안정성 모듈 - 성능 핵심이므로 완전 유지"""
    def __init__(self, channels):
        super().__init__()
        # Fine-grained processing for small patterns
        self.fine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3)
        )
        
        # Stable gradient flow
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        out = self.fine_conv(x)
        out = self.dropout(out)
        return F.relu(x + out)

class LiteEnhancedMultiScaleModule(nn.Module):
    """경량화된 Multi-scale 모듈 - 2-branch로 축소"""
    def __init__(self, channels):
        super().__init__()
        # 핵심 2개 branch만 유지
        self.branch1 = nn.Conv2d(channels, channels//2, kernel_size=1, bias=False)
        self.branch2 = SeparableConv2d(channels, channels//2, kernel_size=3, padding=2, dilation=2, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        concat = torch.cat([b1, b2], dim=1)
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

# --- MicroNetv9 인코더 (Ultra-Lite) ---

class MicroNetV9Encoder(nn.Module):
    """MicroNetv9 인코더 - Ultra-Lite (10K 파라미터 목표)"""
    def __init__(self, in_channels):
        super().__init__()

        # 극도 경량화 채널: 3 → 8 → 12 → 16
        self.downsample_1 = MicroDownsampleModule(in_channels, 8)
        self.downsample_2 = MicroDownsampleModule(8, 12)
        
        # 핵심 모듈만 유지
        self.downsample_modules = nn.Sequential(
            CFDStabilizedModule(12),
            LiteEnhancedMultiScaleModule(12)
        )
        
        self.downsample_3 = MicroDownsampleModule(12, 16)

        # Feature modules: 3개로 축소 (rates: 1,2,4)
        rates = [1, 2, 4]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(16, rate, 0.08) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection

# --- 최종 제출 모델: MicroNetv9 (Ultra-Lite) ---
class submission_MicroNetv9(nn.Module):
    """MicroNetv9 - Ultra-Lite (10K 파라미터 목표)"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (Ultra-Lite)
        self.encoder = MicroNetV9Encoder(in_channels)

        # 간소화된 Auxiliary path
        self.aux_downsample = MicroDownsampleModule(in_channels, 8)
        self.aux_refine = CFDStabilizedModule(8)

        # 간소화된 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(16, 12)
        
        # 최소한의 upsample processing
        self.upsample_mods = CFDStabilizedModule(12)

        # 출력
        self.output_conv = nn.ConvTranspose2d(12, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path (저수준 특징 보존)
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        
        # Main encoder
        enc, skip = self.encoder(x)
        
        # 간소화된 decoder
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
    model = submission_MicroNetv9(3, 21)
    print(f"Total parameters: {model.count_parameters():,}")
    
    # 테스트
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 목표 대비 분석
    target = 10000
    total_params = model.count_parameters()
    if total_params <= target:
        print(f'✅ 목표 달성! ({target:,}개 이내)')
    else:
        print(f'❌ 목표 초과 ({total_params - target:,}개 초과)') 