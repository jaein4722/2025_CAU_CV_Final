import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules (v5와 동일) ---

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

# --- v7_lite 경량 모듈들 ---

class LiteAdaptiveASPP(nn.Module):
    """경량화된 Adaptive ASPP - 파라미터 최소화"""
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 경량화: 모든 경우에 동일한 구조 사용 (차이는 최소화)
        mid_channels = max(3, out_channels // 5)  # 더 작은 채널
        
        # Multi-class일 때만 dilation 1개 추가
        if num_classes > 2:
            dilations = [1, 6, 12]  # 3개로 줄임
        else:
            dilations = [1, 6]      # 2개만
        
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(DWSConv(in_channels, mid_channels, 1, 1, padding=0, bn_acti=True))
            else:
                self.branches.append(DWSConv(in_channels, mid_channels, 3, 1, padding=d, dilation=(d, d), bn_acti=True))
        
        # Global pooling 경량화
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.Sigmoid()  # BN + activation 제거
        )
        
        # Fusion 경량화
        total_channels = mid_channels * (len(dilations) + 1)
        self.fusion = nn.Conv2d(total_channels, out_channels, 1, bias=False)  # DWSConv → Conv로 변경
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)
        
    def forward(self, x):
        outputs = []
        
        # Multi-scale features
        for branch in self.branches:
            outputs.append(branch(x))
        
        # Global context
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        outputs.append(gp)
        
        # Concatenate and fuse
        concat = torch.cat(outputs, dim=1)
        out = self.fusion(concat)
        return self.acti(self.bn(out))

class LiteClassAttention(nn.Module):
    """경량화된 Class-Aware Attention"""
    def __init__(self, channels, num_classes, reduction=24):  # reduction 증가
        super().__init__()
        self.num_classes = num_classes
        
        # Multi-class여도 채널 어텐션만 사용 (공간 어텐션 제거)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention만 적용 (공간 어텐션 완전 제거)
        ca = self.channel_attention(x)
        return x * ca

class LitePyramidDecoder(nn.Module):
    """경량화된 Pyramid Decoder"""
    def __init__(self, high_channels, low_channels, out_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Lateral connection 경량화
        self.lateral = nn.Conv2d(low_channels, out_channels, 1, bias=False)  # Conv로 단순화
        self.lateral_bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        
        # Top-down pathway
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Feature refinement 단순화 (모든 경우에 1-layer만)
        self.refine = DWSConv(high_channels + out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        
    def forward(self, high_feat, low_feat):
        # Upsample high-level features
        high_up = self.up(high_feat)
        
        # Process low-level features (경량화)
        low_lateral = F.relu(self.lateral_bn(self.lateral(low_feat)), inplace=True)
        
        # Combine features
        combined = torch.cat([high_up, low_lateral], dim=1)
        
        # Refine combined features
        return self.refine(combined)

# --- 최종 제출 모델: HWNetUltra_v7_lite (극도 경량화) ---
class submission_HWNetUltra_v7_lite(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # v5 기반 설정 유지
        block_1 = 2
        block_2 = 2
        C = 8
        P = 0.25
        dilation_block_1 = [2, 3]
        dilation_block_2 = [2, 4]

        # 엣지 강화 모듈 (v5와 동일)
        self.edge_focus = MorphGradientFocus(in_channels)
        
        # 초기 특징 추출 (v5와 동일)
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        # 1단계 인코더 (v5와 동일)
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        # 2단계 인코더 (v5와 동일)
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))
        
        # 🆕 v7_lite: 경량화된 Adaptive ASPP
        self.aspp = LiteAdaptiveASPP(C * 4, C * 4, num_classes)
        
        # 🆕 v7_lite: 경량화된 Class-Aware Attention
        self.attention = LiteClassAttention(C * 4, num_classes, reduction=24)
        
        # 🆕 v7_lite: 경량화된 Pyramid Decoder
        self.dec2 = LitePyramidDecoder(C * 4, C * 2, C * 2, num_classes)
        self.dec1 = LitePyramidDecoder(C * 2, C, C, num_classes)
        
        # 최종 출력 경량화 (동일한 구조)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True),
            DWSConv(C, C//2, 3, 1, padding=1, bn_acti=True),
            nn.Conv2d(C//2, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # 엣지 강화
        x_init = self.edge_focus(input)
        
        # 인코더
        x0 = self.Init_Block(x_init)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        
        # 🆕 v7_lite: 경량화된 multi-scale context
        x2 = self.aspp(x2)
        
        # 🆕 v7_lite: 경량화된 class-aware attention
        x2 = self.attention(x2)
        
        # 🆕 v7_lite: 경량화된 pyramid decoder
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        
        # 최종 출력
        return self.final_up(d1)

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    for num_classes in [2, 21]:
        net = submission_HWNetUltra_v7_lite(in_channels=3, num_classes=num_classes)
        p = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Classes: {num_classes}, Params: {p:,}")
        
        # 파라미터 목표 검증
        if p < 10000:
            print(f"✅ 목표 달성: {p}/10,000 ({10000-p} 여유)")
        elif p <= 17000:
            print(f"⚠️ 허용 범위: {p}/17,000 ({17000-p} 여유)")
        else:
            print(f"❌ 초과: {p}/17,000 ({p-17000} 초과)")

        try:
            net.eval()
            x = torch.randn(1, 3, 256, 256)
            y = net(x)
            print(f"Input: {x.shape} → Output: {y.shape}")
            assert y.shape == (1, num_classes, 256, 256)
            print("✅ 테스트 통과\n")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}\n")
    
    print(f"📊 HWNetUltra_v7_lite 경량화 개선:")
    print(f"  💡 LiteAdaptiveASPP: 채널 최소화 + dilation 차별화")
    print(f"  💡 LiteClassAttention: Channel attention만 (spatial 제거)")  
    print(f"  💡 LitePyramidDecoder: 단순화된 feature fusion")
    print(f"  💡 Unified Final Layers: 모든 클래스에 동일 구조")
    print(f"  🎯 목표: 1만개 미만 파라미터로 IoU 0.43+ 달성") 