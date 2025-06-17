import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MicroNetv13_Optimized: 경량화된 최종 모델 ---

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
    """Multi-dilation separable conv - 성능 핵심 모듈"""
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

# --- test 모델의 핵심 성공 요소들 (경량화) ---

class ECA(nn.Module):
    """Efficient Channel Attention - test 모델의 핵심"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention via 1D convolution
        y = self.avg_pool(x).view(x.size(0), 1, -1)
        y = self.conv(y)
        y = self.sigmoid(y).view(x.size(0), -1, 1, 1)
        return x * y

class SimpleASPP(nn.Module):
    """Simple ASPP - 경량화된 2-branch ASPP"""
    def __init__(self, in_channels, mid_channels=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 2-branch ASPP (경량화)
        self.branch1 = SeparableConv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = SeparableConv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        
        # Projection
        self.project = nn.Conv2d(mid_channels * 3, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        # Global pooling branch
        pool = F.interpolate(self.pool_branch(self.global_pool(x)), size=(h, w), mode='bilinear', align_corners=False)
        
        # Multi-scale branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        # Concatenate and project
        concat = torch.cat([pool, b1, b2], dim=1)
        out = self.project(concat)
        return F.relu(out)

class MorphGradientFocus(nn.Module):
    """Morphological Gradient Focus - test 모델의 Edge enhancement"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Compute gradient via morphological operations
        gray = x.mean(dim=1, keepdim=True)
        dilation = F.max_pool2d(gray, kernel_size=3, stride=1, padding=self.pad)
        erosion = -F.max_pool2d(-gray, kernel_size=3, stride=1, padding=self.pad)
        gradient = dilation - erosion
        
        # Fuse with original features
        return self.fuse(torch.cat([x, gradient], dim=1))

class OptimizedCFDModule(nn.Module):
    """Optimized CFD 모듈 - 경량화된 CFD 처리"""
    def __init__(self, channels):
        super().__init__()
        # Enhanced processing for crack detection
        self.fine_conv = SeparableConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x):
        out = self.fine_conv(x)
        out = self.bn(out)
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

class ResidualConvModule(nn.Module):
    """Residual Conv 모듈 - test 모델 스타일"""
    def __init__(self, channels, dilation=1, dropout=0.0):
        super().__init__()
        self.conv = SeparableConv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
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

# --- MicroNetv13_Optimized 인코더 ---

class MicroNetV13OptimizedEncoder(nn.Module):
    """MicroNetv13_Optimized 인코더 - 경량화된 설계"""
    def __init__(self, in_channels):
        super().__init__()

        # 경량화된 채널: 3 → 6 → 12 → 18
        self.downsample_1 = MicroDownsampleModule(in_channels, 6)
        self.downsample_2 = MicroDownsampleModule(6, 12)
        
        # 핵심 모듈들 (경량화)
        self.cfd_module = OptimizedCFDModule(12)
        self.mgf_module = MorphGradientFocus(12)  # Edge enhancement
        
        self.downsample_3 = MicroDownsampleModule(12, 18)

        # Simple ASPP - 경량화
        self.aspp = SimpleASPP(18, mid_channels=4)
        
        # ECA Attention - test 모델의 핵심
        self.eca = ECA(18, k_size=3)

        # Feature modules: 2개만 (rates: 1,2)
        rates = [1, 2]
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(18, rate, 0.1) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        
        # 핵심 모듈 적용
        d2 = self.cfd_module(d2)
        d2 = self.mgf_module(d2)  # Edge enhancement
        
        d3 = self.downsample_3(d2)
        
        # ASPP + ECA 적용 (test 모델 스타일)
        d3 = self.aspp(d3)
        d3 = self.eca(d3)
        
        m4 = self.feature_modules(d3)
        
        return m4, d2  # skip connection

# --- 최종 제출 모델: MicroNetv13_Optimized ---
class submission_MicroNetv13(nn.Module):
    """MicroNetv13_Optimized - 경량화된 최종 모델 (Mean IoU 0.45+ 목표)"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (Optimized)
        self.encoder = MicroNetV13OptimizedEncoder(in_channels)

        # 업샘플 블록
        self.upsample_1 = MicroUpsampleModule(18, 12)
        
        # 2-way Fusion (경량화)
        self.fusion = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1, bias=False),  # 12 + 12 = 24
            nn.BatchNorm2d(12, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Processing blocks (경량화: 2개만)
        self.up_blocks = nn.Sequential(*[
            ResidualConvModule(12, dilation=1, dropout=0.05) for _ in range(2)
        ])

        # 출력 (bias=True for final layer)
        self.output_conv = nn.ConvTranspose2d(12, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Main encoder
        enc, skip = self.encoder(x)
        
        # Decoder
        up = self.upsample_1(enc)
        
        # 2-way Fusion: up + skip
        if up.shape[2:] == skip.shape[2:] and up.shape[1] == skip.shape[1]:
            fused = self.fusion(torch.cat([up, skip], dim=1))
        else:
            # Fallback
            fused = up
            fused = self.fusion(torch.cat([fused, fused], dim=1))  # Dummy 2-way
        
        # Processing
        fused = self.up_blocks(fused)
        
        # 최종 출력
        output = self.output_conv(fused)
        
        if self.interpolate and output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output

    def count_parameters(self):
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 파라미터 수 확인용
if __name__ == "__main__":
    # VOC 기준 테스트 (21 클래스)
    model_voc = submission_MicroNetv13(3, 21)
    params_voc = model_voc.count_parameters()
    print(f"VOC (21 classes) parameters: {params_voc:,}")
    
    # Binary 기준 테스트 (2 클래스)
    model_binary = submission_MicroNetv13(3, 2)
    params_binary = model_binary.count_parameters()
    print(f"Binary (2 classes) parameters: {params_binary:,}")
    
    # 목표 대비 분석
    target = 9000
    if abs(params_voc - target) <= 1000:
        print(f'✅ 목표 달성! ({target:,}개 근처)')
        diff = params_voc - target
        print(f'차이: {diff:+,}개')
    else:
        print(f'❌ 목표 미달성 ({abs(params_voc - target):,}개 차이)')
    
    # 테스트
    x = torch.randn(1, 3, 256, 256)
    try:
        y = model_voc(x)
        print(f"✅ 모델 테스트 성공: {x.shape} → {y.shape}")
    except Exception as e:
        print(f"❌ 모델 테스트 실패: {e}")
    
    print(f"\n🏆 Optimized 전략:")
    print(f"v13_Ultimate: 18,606개 → v13_Optimized: {params_voc:,}개")
    reduction = 18606 - params_voc
    print(f"절약량: {reduction:,}개 ({reduction/18606*100:.1f}% 절약)")
    
    print(f"\n🎯 최종 목표:")
    print(f"- Mean IoU: 0.45+ (test 모델 수준)")
    print(f"- 파라미터: ~9,000개")
    print(f"- test 모델 핵심 요소 유지")
    
    print(f"\n🚀 Optimized 핵심사항:")
    print(f"- ECA Attention (test 모델 핵심)")
    print(f"- MorphGradientFocus (Edge enhancement)")
    print(f"- OptimizedCFDModule (CFD 성능)")
    print(f"- Simple ASPP (2-branch)")
    print(f"- 2-way Fusion (up + skip)")
    print(f"- 2개 ResidualConvModule")
    print(f"- 경량화된 채널 (6→12→18)") 