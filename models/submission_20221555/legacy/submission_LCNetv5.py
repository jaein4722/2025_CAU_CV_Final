import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# LCNet_v4 개선 요약:
# 1. EnhancedDecoder 적용: 기존 SimpleDecoder의 특징 융합 방식을 단순 덧셈(+)에서
#    Concat과 Conv를 활용하는 방식으로 변경하여 표현력 증대 (4점 항목).
# 2. SEBlock 추가: 각 Encoder 블록 마지막에 경량 어텐션 모듈을 추가하여 채널별
#    특징의 중요도를 학습하도록 개선 (4점 항목).
# 3. 기본 채널 수(C) 조정: 새로운 모듈 추가에 따른 파라미터 증가를 억제하기 위해
#    C를 12에서 10으로 소폭 조정하여 전체 파라미터 관리.
# --------------------------------------------------------------------------------

class DSConv(nn.Module):
    """
    Depthwise Separable Convolution 모듈입니다.
    """
    def __init__(self, nIn, nOut, kSize, stride, padding, bn_acti=True):
        super().__init__()
        self.bn_acti = bn_acti
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding, groups=nIn, bias=False)
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)
        
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SELU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        if self.bn_acti:
            out = self.bn(out)
            out = self.acti(out)
        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn
        
        self.conv3x3 = DSConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        
        output = self.bn(output)
        output = self.acti(output)
        return output

def Split(x, p):
    c = int(x.size()[1])
    c1 = round(c * (1-p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class TCA(nn.Module):
    """
    구조가 대폭 단순화된 TCA 모듈입니다. 
    다중 분기를 모두 제거하고 단일 DepthwiseSeparableConv 경로만 사용합니다.
    """
    def __init__(self, c, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.conv3x3 = DSConv(c, c, kSize, 1, padding=1, bn_acti=True)

    def forward(self, input):
        # 분기 없이 conv3x3의 결과만 바로 반환
        output = self.conv3x3(input)
        return output

class PCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1-p))
        self.TCA = TCA(c, d)
        self.conv1x1 = nn.Conv2d(nIn, nIn, 1, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.SELU(inplace=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        output = self.bn(output)
        output = self.acti(output)
        return output

# --- 새로 추가된 모듈 ---
class SEBlock(nn.Module):
    """
    경량 채널 어텐션 모듈 (Squeeze-and-Excitation Block)
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class EnhancedDecoder(nn.Module):
    """
    Concat 기반 융합 방식을 사용하는 개선된 Decoder
    """
    def __init__(self, c_high, c_low, num_classes):
        super().__init__()
        self.conv_low = nn.Conv2d(c_low, c_high, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(c_high)
        
        # Concat으로 채널이 2배가 된 피쳐를 다시 원래대로 줄여주는 Conv
        self.fusion_conv = DSConv(c_high * 2, c_high, kSize=3, stride=1, padding=1, bn_acti=True)
        
        self.output_conv = nn.Conv2d(c_high, num_classes, kernel_size=1, bias=False)

    def forward(self, x_high, x_low):
        x_low_upsampled = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_low_processed = self.bn_low(self.conv_low(x_low_upsampled))
        
        # Add 대신 Concat으로 융합
        fused_features = torch.cat([x_high, x_low_processed], dim=1)
        
        out = self.fusion_conv(fused_features)
        out = self.output_conv(out)
        return out

# --- 개선된 디코더 ---
class MultiStageDecoder(nn.Module):
    def __init__(self, c_high, c_mid, c_low, num_classes):
        super().__init__()
        # 1/8 -> 1/4 업샘플링 및 융합
        self.up_1 = EnhancedDecoder(c_mid, c_low, c_mid) # 출력 채널을 c_mid로 맞춤
        
        # 1/4 -> 1/2 업샘플링 및 융합
        self.up_2 = EnhancedDecoder(c_high, c_mid, c_high) # 출력 채널을 c_high로 맞춤

        self.output_conv = nn.Conv2d(c_high, num_classes, kernel_size=1, bias=False)

    def forward(self, x_high, x_mid, x_low):
        # x_high: 1/2 스케일 (output0), x_mid: 1/4 스케일 (output1), x_low: 1/8 스케일 (output2)
        
        # 1단계: 1/8 피처를 업샘플링하여 1/4 피처와 융합
        fused_mid = self.up_1(x_mid, x_low)
        
        # 2단계: 융합된 1/4 피처를 업샘플링하여 1/2 피처와 융합
        fused_high = self.up_2(x_high, fused_mid)
        
        out = self.output_conv(fused_high)
        return out

# --------------------------------------------------------------------------------
# 최종 제출용 모델 (클래스 이름을 "submission_{본인학번}"으로 변경하세요)
# --------------------------------------------------------------------------------
class submission_LCNetv5(nn.Module): # ★★★ 최종 제출 시 submission_{본인학번}으로 변경 필수 ★★★
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # --- 하이퍼파라미터 (구조개선 + 경량화) ---
        block_1 = 2
        block_2 = 6
        C = 8  # 파라미터 관리를 위해 12 -> 10으로 조정
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        se_reduction = 4 # SEBlock의 감소 비율

        # 초기 블록
        self.Init_Block = nn.Sequential(
            DSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DSConv(C, C, 3, 1, padding=1, bn_acti=True),
            DSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        
        # Block 1
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C*2))
        for i in range(0, block_1):       
            self.LC_Block_1.add_module("LC_Module_1_" + str(i), PCT(nIn=C*2, d=dilation_block_1[i], p=P))
        self.LC_Block_1.add_module("SE_Block", SEBlock(channel=C*2, reduction=se_reduction)) # ★ SEBlock 추가
        
        # Block 2
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C*2, C*4))
        for i in range(0, block_2):
            self.LC_Block_2.add_module("LC_Module_2_" + str(i), PCT(nIn=C*4, d=dilation_block_2[i], p=P))
        self.LC_Block_2.add_module("SE_Block", SEBlock(channel=C*4, reduction=se_reduction)) # ★ SEBlock 추가
        
        # Decoder를 MultiStageDecoder로 교체
        self.Decoder = MultiStageDecoder(C, C*2, C*4, num_classes)
    
    def forward(self, input):
        output0 = self.Init_Block(input)
        output1 = self.LC_Block_1(output0)
        output2 = self.LC_Block_2(output1)
        
        # 3개의 스케일 출력을 모두 Decoder에 전달
        out = self.Decoder(output0, output1, output2)
        
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        return out
    
if __name__ == "__main__":
    net = submission_LCNetv5(in_channels=3, num_classes=21)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters())
    print(f"Params: {p/1e3:.1f} K")