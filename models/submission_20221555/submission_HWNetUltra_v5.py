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
    def __init__(self, nIn, d=1, p=0.25):
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

# --- v5 새로운 모듈들 (최적화 버전) ---

class LightASPP(nn.Module):
    """경량 ASPP - 파라미터 최적화 버전"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3개 branch로 축소 + 적절한 채널 크기
        mid_channels = max(4, out_channels // 4)  # 최소 4채널 보장
        
        self.branch1 = DWSConv(in_channels, mid_channels, 1, 1, padding=0, bn_acti=True)  # 1x1
        self.branch2 = DWSConv(in_channels, mid_channels, 3, 1, padding=6, dilation=(6, 6), bn_acti=True)   # 3x3, d=6
        self.branch3 = DWSConv(in_channels, mid_channels, 3, 1, padding=12, dilation=(12, 12), bn_acti=True) # 3x3, d=12
        
        # Global pooling branch (더 경량화)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1e-3),
            nn.SiLU(inplace=True)
        )
        
        # Feature fusion (더 경량화)
        self.fusion = DWSConv(mid_channels * 4, out_channels, 1, 1, padding=0, bn_acti=True)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Global pooling and upsample
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        concat = torch.cat([b1, b2, b3, gp], dim=1)
        return self.fusion(concat)

class SCAM(nn.Module):
    """Spatial-Channel Attention Module - 경량화 버전"""
    def __init__(self, channels, reduction=16):  # reduction 증가
        super().__init__()
        # Channel Attention만 유지 (Spatial attention 제거로 파라미터 절약)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention만 적용
        ca = self.channel_attention(x)
        return x * ca

class PyramidDecoder(nn.Module):
    """경량 피라미드 디코더"""
    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        # Lateral connection 단순화
        self.lateral = Conv(low_channels, out_channels, 1, 1, padding=0, bn_acti=True)
        
        # Top-down pathway
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Feature refinement 단순화 (1개 레이어만)
        self.refine = DWSConv(high_channels + out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        
    def forward(self, high_feat, low_feat):
        # Upsample high-level features
        high_up = self.up(high_feat)
        
        # Process low-level features
        low_lateral = self.lateral(low_feat)
        
        # Combine features
        combined = torch.cat([high_up, low_lateral], dim=1)
        
        # Refine combined features
        return self.refine(combined)

# --- 최종 제출 모델: HWNetUltra_v5 (Multi-scale Context + Enhanced Decoder) ---
class submission_HWNetUltra_v5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # v4 기반 설정 유지
        block_1 = 2
        block_2 = 2
        C = 8
        P = 0.25
        dilation_block_1 = [2, 3]
        dilation_block_2 = [2, 4]

        # 엣지 강화 모듈 (v4와 동일)
        self.edge_focus = MorphGradientFocus(in_channels)
        
        # 초기 특징 추출 (v4와 동일)
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        # 1단계 인코더 (v4와 동일)
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        # 2단계 인코더 (v4와 동일)
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        # 🆕 Multi-scale Context Module (ASPP)
        self.aspp = LightASPP(C * 4, C * 4)
        
        # 🆕 Spatial-Channel Attention
        self.attention = SCAM(C * 4, reduction=16)
        
        # 🆕 Enhanced Pyramid Decoder
        self.dec2 = PyramidDecoder(C * 4, C * 2, C * 2)
        self.dec1 = PyramidDecoder(C * 2, C, C)
        
        # 최종 출력 (개선된 버전)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True),
            DWSConv(C, C//2, 3, 1, padding=1, bn_acti=True),  # 추가 refinement
            nn.Conv2d(C//2, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # 엣지 강화
        x_init = self.edge_focus(input)
        
        # 인코더
        x0 = self.Init_Block(x_init)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        
        # 🆕 Multi-scale context 추출
        x2 = self.aspp(x2)
        
        # 🆕 Spatial-Channel attention 적용
        x2 = self.attention(x2)
        
        # 🆕 Enhanced pyramid decoder
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        
        # 최종 출력
        return self.final_up(d1)

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_HWNetUltra_v5(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNetUltra_v5 (Multi-scale Context + Enhanced Decoder)")
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
        net.eval()  # 테스트 모드로 설정
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("✅ 모델 실행 테스트 통과")
        
        # 다양한 클래스 수에 대한 테스트
        for nc in [1, 2, 21]:
            net_test = submission_HWNetUltra_v5(in_channels=3, num_classes=nc)
            net_test.eval()  # 테스트 모드로 설정
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        
    # v4와 비교 분석
    print(f"\n📊 v4 대비 주요 개선사항:")
    print(f"  🆕 LightASPP: Multi-scale context (VOC 성능 개선)")
    print(f"  🆕 SCAM: Spatial-Channel attention")
    print(f"  🆕 PyramidDecoder: FPN 스타일 feature fusion")
    print(f"  🆕 Enhanced final layers: 추가 refinement")
    print(f"  파라미터: 7,150 → {p:,} (+{p-7150:,})")
    print(f"  증가율: +{(p-7150)/7150*100:.1f}%")
    print(f"  목표: VOC 성능 대폭 개선 + 전체 IoU 0.42+ 달성")
    
    # 새로운 모듈별 파라미터 분석
    aspp_params = sum(p.numel() for p in net.aspp.parameters())
    attention_params = sum(p.numel() for p in net.attention.parameters())
    dec2_params = sum(p.numel() for p in net.dec2.parameters())
    dec1_params = sum(p.numel() for p in net.dec1.parameters())
    
    print(f"\n🎯 새로운 모듈별 파라미터:")
    print(f"  LightASPP: {aspp_params:,} ({aspp_params/p*100:.1f}%)")
    print(f"  SCAM: {attention_params:,} ({attention_params/p*100:.1f}%)")
    print(f"  PyramidDecoder-2: {dec2_params:,} ({dec2_params/p*100:.1f}%)")
    print(f"  PyramidDecoder-1: {dec1_params:,} ({dec1_params/p*100:.1f}%)")
    print(f"  총 추가: {aspp_params+attention_params+dec2_params+dec1_params:,}") 