import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules ---

class DWSConv(nn.Module):
    """Depthwise Separable Convolution - 파라미터 효율성을 위한 핵심 모듈"""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, dilation=(1, 1), bn_acti=True, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding,
                                   dilation=dilation, groups=nIn, bias=bias)
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_acti = bn_acti
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_acti:
            x = self.bn(x)
            x = self.acti(x)
        return x

class Conv(nn.Module):
    """1x1 Conv나 Grouped Conv를 위한 표준 모듈"""
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn(output)
            output = self.acti(output)
        return output

class DownSamplingBlock(nn.Module):
    """효율적인 다운샘플링 - HWNet 정체성 유지"""
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn, self.nOut = nIn, nOut
        nConv = nOut - nIn if self.nIn < self.nOut else nOut
        self.conv3x3 = DWSConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            output = torch.cat([output, self.max_pool(input)], 1)
        return self.acti(self.bn(output))

def Split(x, p):
    """채널 분할 - HWNet/LCNet의 핵심 정체성"""
    c = int(x.size(1))
    c1 = round(c * (1 - p))
    return x[:, :c1, :, :].contiguous(), x[:, c1:, :, :].contiguous()

class TCA(nn.Module):
    """Triple Context Aggregation - 다중 스케일 특징 추출"""
    def __init__(self, c, d=1, kSize=3):
        super().__init__()
        self.conv3x3 = DWSConv(c, c, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x3 = Conv(c, c, (kSize, kSize), 1, padding=(1, 1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c, c, (kSize, kSize), 1, padding=(d, d), groups=c, dilation=(d, d), bn_acti=True)
        self.bn = nn.BatchNorm2d(c, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        br = self.conv3x3(input)
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        return self.acti(self.bn(br))

class PCT(nn.Module):
    """Partial Channel Transformation - HWNet의 핵심 병목 구조"""
    def __init__(self, nIn, d=1, p=0.25):  # v2와 동일한 p=0.25 유지
        super().__init__()
        self.p = p
        c = nIn - round(nIn * (1 - p))
        self.TCA = TCA(c, d)
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        x1, x2 = Split(input, self.p)
        x2 = self.TCA(x2)
        output = torch.cat([x1, x2], dim=1)
        return self.conv1x1(output) + input

class LightDecoderBlock(nn.Module):
    """경량 디코더 블록"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DWSConv(in_channels + skip_channels, out_channels, 3, 1, padding=1, bn_acti=True),
            DWSConv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class MorphGradientFocus(nn.Module):
    """모폴로지 기반 엣지 강화 - 파라미터 효율적"""
    def __init__(self, in_channels, k=3):
        super().__init__()
        self.pad  = k // 2
        self.fuse = Conv(in_channels + 1, in_channels, 1, 1, padding=0, bn_acti=True)

    def forward(self, x):
        intensity = x.mean(dim=1, keepdim=True)
        dilated = F.max_pool2d(intensity, 3, stride=1, padding=self.pad)
        eroded  = -F.max_pool2d(-intensity, 3, stride=1, padding=self.pad)
        return self.fuse(torch.cat([x, dilated - eroded], dim=1))

class StableSE(nn.Module):
    """Stable Squeeze-and-Excitation - v2 기반 안정적 어텐션 강화"""
    def __init__(self, channels, reduction=12):  # reduction을 16→12로 감소 (유일한 변경점)
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- 최종 제출 모델: HWNetUltra_v4 (안정적 성능 향상) ---
class submission_HWNetUltra_v4(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # v2 기반 안정적 설정 (SE attention만 강화)
        block_1 = 2  # v2와 동일
        block_2 = 2  # v2와 동일
        C = 8        # v2와 동일 (안정성 우선)
        P = 0.25     # v2와 동일 (검증된 비율)
        dilation_block_1 = [2, 3]  # v2와 동일
        dilation_block_2 = [2, 4]  # v2와 동일

        # 엣지 강화 모듈 (v2와 동일)
        self.edge_focus = MorphGradientFocus(in_channels)
        
        # 초기 특징 추출 (v2와 동일 - 2레이어)
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        # 1단계 인코더 (v2와 동일)
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        # 2단계 인코더 (v2와 동일)
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        # 강화된 어텐션 (유일한 변경점: reduction 16→12)
        self.attention = StableSE(C * 4, reduction=12)
        
        # 디코더 (v2와 동일)
        self.dec2 = LightDecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = LightDecoderBlock(C * 2, C, C)
        
        # 최종 출력 (v2와 동일)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True),
            nn.Conv2d(C, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # 엣지 강화 (v2와 동일)
        x_init = self.edge_focus(input)
        
        # 인코더 (v2와 동일)
        x0 = self.Init_Block(x_init)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        
        # 강화된 어텐션 적용 (유일한 변경점)
        x2 = self.attention(x2)
        
        # 디코더 (v2와 동일)
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        
        # 최종 출력 (v2와 동일)
        return self.final_up(d1)

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_HWNetUltra_v4(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNetUltra_v4 (안정적 성능 향상)")
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
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("✅ 모델 실행 테스트 통과")
        
        # 다양한 클래스 수에 대한 테스트
        for nc in [1, 2, 21]:
            net_test = submission_HWNetUltra_v4(in_channels=3, num_classes=nc)
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # v2와 비교 분석
    print(f"\n📊 v2 대비 변화:")
    print(f"  구조: 동일 (C=8, p=0.25, 2-layer init)")
    print(f"  변경점: SE reduction 16→12 (어텐션 강화)")
    print(f"  파라미터: 7,150 → {p:,} (+{p-7150:,})")
    print(f"  증가율: +{(p-7150)/7150*100:.1f}% (최소한 증가)")
    print(f"  목표: ETIS 성능 회복 + 전체 IoU 0.40+ 달성")
    
    # 어텐션 파라미터 분석
    attention_params = sum(p.numel() for p in net.attention.parameters())
    print(f"\n🎯 어텐션 모듈:")
    print(f"  파라미터: {attention_params:,}")
    print(f"  전체 대비: {attention_params/p*100:.1f}%")
    print(f"  reduction: 12 (v2: 16, v3: 14)") 