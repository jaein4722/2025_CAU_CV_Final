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
    """Triple Context Aggregation - 최적화된 다중 스케일 특징 추출"""
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
    def __init__(self, nIn, d=1, p=0.25):  # p를 0.2→0.25로 증가하여 파라미터 절약
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

class UltraLightSE(nn.Module):
    """Ultra Light Squeeze-and-Excitation - 극도로 경량화된 어텐션"""
    def __init__(self, channels, reduction=16):  # reduction을 8→16으로 증가
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

# --- 최종 제출 모델: HWNetUltra_v2 (파라미터 최적화 버전) ---
class submission_HWNetUltra_v2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # 1만 미만 파라미터를 위한 최적화된 설정
        block_1 = 2  # 유지
        block_2 = 2  # 3→2로 감소 (파라미터 절약)
        C = 8        # 10→8로 감소 (주요 파라미터 절약)
        P = 0.25     # 0.2→0.25로 증가 (TCA 채널 감소)
        dilation_block_1 = [2, 3]
        dilation_block_2 = [2, 4]  # [2,4,8]→[2,4]로 단순화

        # 엣지 강화 모듈 (성능 핵심 - 유지)
        self.edge_focus = MorphGradientFocus(in_channels)
        
        # 초기 특징 추출 (2레이어로 단순화)
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        # 1단계 인코더
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        # 2단계 인코더 (블록 수 감소)
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        # 극경량 어텐션 (reduction 증가로 파라미터 대폭 절약)
        self.attention = UltraLightSE(C * 4, reduction=16)
        
        # 디코더
        self.dec2 = LightDecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = LightDecoderBlock(C * 2, C, C)
        
        # 최종 출력
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True),
            nn.Conv2d(C, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # 엣지 강화 (성능 핵심 유지)
        x_init = self.edge_focus(input)
        
        # 인코더
        x0 = self.Init_Block(x_init)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        
        # 어텐션 적용
        x2 = self.attention(x2)
        
        # 디코더
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        
        # 최종 출력
        return self.final_up(d1)

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_HWNetUltra_v2(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNetUltra_v2 (파라미터 최적화 버전)")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 1만 미만 목표 검증
    if p < 10000:
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
            net_test = submission_HWNetUltra_v2(in_channels=3, num_classes=nc)
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # 파라미터 분석
    print("\n📊 파라미터 분석:")
    total_params = 0
    for name, module in net.named_modules():
        if len(list(module.children())) == 0:  # leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                total_params += params
                print(f"  {name}: {params:,}")
    print(f"Total: {total_params:,}") 