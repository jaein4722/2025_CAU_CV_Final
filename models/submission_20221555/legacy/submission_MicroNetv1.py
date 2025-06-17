import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MiniNetv2 핵심 모듈들 (경량화 버전) ---

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution - MiniNetv2의 핵심"""
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
    """Multi-dilation separable conv - MiniNetv2의 성능 핵심"""
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
    """경량화된 다운샘플링 모듈"""
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
    """경량화된 Residual 모듈"""
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
    """경량화된 Multi-dilation Residual 모듈"""
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
    """경량화된 업샘플링 모듈"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)

# --- MicroNetv1 인코더 (극도 경량화) ---

class MicroNetV1Encoder(nn.Module):
    """MicroNetv1 인코더 - 518K → 10K 도전"""
    def __init__(self, in_channels):
        super().__init__()

        # 채널 수 극도 감소: 3 → 8 → 16 → 20 (vs 원본 3 → 16 → 64 → 128)
        self.downsample_1 = MicroDownsampleModule(in_channels, 8)
        self.downsample_2 = MicroDownsampleModule(8, 16)
        
        # Downsample modules: 10개 → 1개
        self.downsample_modules = MicroResidualConvModule(16, 1, 0)
        
        self.downsample_3 = MicroDownsampleModule(16, 20)

        # Feature modules: 16개 → 2개, 핵심 dilation rates만 유지
        rates = [1, 2]  # vs 원본 [1,2,1,4,1,8,1,16,1,1,1,2,1,4,1,8]
        self.feature_modules = nn.Sequential(*[MicroResidualMultiDilationConvModule(20, rate, 0.1) for rate in rates])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m1 = self.downsample_modules(d2)
        d3 = self.downsample_3(m1)
        m4 = self.feature_modules(d3)
        return m4

# --- 최종 제출 모델: MicroNetv1 (극도 경량화) ---
class submission_MicroNetv1(nn.Module):
    """MicroNetv1 - MiniNetv2의 98% 경량화 버전"""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 인코더 (극도 경량화)
        self.encoder = MicroNetV1Encoder(in_channels)

        # Auxiliary path 제거 (파라미터 절약)

        # 업샘플 블록 (단순화)
        self.upsample_1 = MicroUpsampleModule(20, 16)
        
        # Upsample modules: 4개 → 1개
        self.upsample_mods = MicroResidualConvModule(16, 1, 0)

        # 출력 (경량화)
        self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        # Auxiliary path 제거로 단순화
        enc = self.encoder(x)
        up1 = self.upsample_1(enc)
        m2 = self.upsample_mods(up1)

        out = self.output_conv(m2)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_MicroNetv1(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MicroNetv1 (MiniNetv2의 98% 경량화)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 파라미터 목표 검증
    if p < 8000:
        print(f"✅ 이상적 범위: {p}/8,000 ({8000-p} 여유)")
    elif p < 10000:
        print(f"✅ 목표 달성: {p}/10,000 ({10000-p} 여유)")
    elif p <= 17000:
        print(f"⚠️  허용 범위 내: {p}/17,000 (hard cap)")
    else:
        print(f"❌ 파라미터 초과: {p}/17,000 ({p-17000} 초과)")

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
            net_test = submission_MicroNetv1(in_channels=3, num_classes=nc)
            net_test.eval()
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # MiniNetv2와 비교 분석
    print(f"\n📊 MiniNetv2 → MicroNetv1 변화:")
    print(f"  파라미터: 518,227 → {p:,} (-{518227-p:,}, -{(518227-p)/518227*100:.1f}%)")
    print(f"  채널 수: 16→64→128 → 8→16→20 (84.4% 감소)")
    print(f"  Feature modules: 16개 → 2개 (87.5% 감소)")
    print(f"  Downsample modules: 10개 → 1개 (90% 감소)")
    print(f"  Auxiliary path: 제거 (파라미터 절약)")
    print(f"  목표: 0.4729 IoU 성능 최대한 유지")
    
    # 모듈별 파라미터 분석
    encoder_params = sum(p.numel() for p in net.encoder.parameters())
    upsample_params = sum(p.numel() for p in net.upsample_1.parameters()) + sum(p.numel() for p in net.upsample_mods.parameters())
    output_params = sum(p.numel() for p in net.output_conv.parameters())
    
    print(f"\n🎯 모듈별 파라미터 분배:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/p*100:.1f}%)")
    print(f"  Upsample: {upsample_params:,} ({upsample_params/p*100:.1f}%)")
    print(f"  Output: {output_params:,} ({output_params/p*100:.1f}%)")
    print(f"  효율성: {p/518227:.4f} (원본 대비 파라미터 비율)")
    
    print(f"\n🚀 MicroNetv1 특징:")
    print(f"  ✅ Multi-dilation 핵심 아이디어 보존")
    print(f"  ✅ Separable convolution 활용")
    print(f"  ✅ Residual connections 유지")
    print(f"  ✅ 98% 파라미터 감소 달성")
    print(f"  🎯 목표: MiniNetv2 성능의 80-90% 유지") 