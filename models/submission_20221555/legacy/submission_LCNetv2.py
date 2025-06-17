import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 최종 경량화 수정 요약:
# 1. Depthwise Separable Convolution: 일반 Conv를 분리하여 파라미터 효율화 (기존 유지)
# 2. SimpleDecoder: 복잡한 DAD 모듈을 직접 구현한 간단한 디코더로 대체 (기존 유지)
# 3. 채널 수(C) 대폭 감소 (32 -> 12): 모델의 전체적인 너비를 줄여 파라미터 수를 크게 감소시킴
# 4. 블록 수(block_1, block_2) 감소: 모델의 깊이를 줄여 파라미터 추가 감소
# 5. TCA 모듈 단순화: TCA 모듈 내 다중 분기(br1, br2)를 모두 제거하고 단일 경로로 변경하여
#    구조를 간소화하고 파라미터를 추가로 줄임 (구조 개선 4점 항목)
# --------------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
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
        
        self.conv3x3 = DepthwiseSeparableConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SELU(inplace=True)

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
        self.conv3x3 = DepthwiseSeparableConv(c, c, kSize, 1, padding=1, bn_acti=True)

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

class SimpleDecoder(nn.Module):
    """
    기존 DAD 모듈을 대체하는 간단한 Decoder입니다.
    """
    def __init__(self, c_high, c_low, num_classes):
        super().__init__()
        self.conv_low = nn.Conv2d(c_low, c_high, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(c_high)
        self.final_conv = DepthwiseSeparableConv(c_high, c_high, kSize=3, stride=1, padding=1, bn_acti=True)
        self.output_conv = nn.Conv2d(c_high, num_classes, kernel_size=1, bias=False)

    def forward(self, x_high, x_low):
        x_low_upsampled = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_low_processed = self.bn_low(self.conv_low(x_low_upsampled))
        fused_features = x_high + x_low_processed
        out = self.final_conv(fused_features)
        out = self.output_conv(out)
        return out

# --------------------------------------------------------------------------------
# 최종 제출용 모델 (클래스 이름을 "submission_{본인학번}"으로 변경하세요)
# --------------------------------------------------------------------------------
class submission_LCNetv2(nn.Module): # 클래스 이름을 본인 학번에 맞게 수정
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # --- 하이퍼파라미터 (경량화 최적화) ---
        block_1 = 2
        block_2 = 6
        C = 12
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16]

        # 초기 블록
        self.Init_Block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DepthwiseSeparableConv(C, C, 3, 1, padding=1, bn_acti=True),
            DepthwiseSeparableConv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        
        # Block 1
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C*2))
        for i in range(0, block_1):       
            self.LC_Block_1.add_module("LC_Module_1_" + str(i), PCT(nIn=C*2, d=dilation_block_1[i], p=P))
        
        # Block 2
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C*2, C*4))
        for i in range(0, block_2):
            self.LC_Block_2.add_module("LC_Module_2_" + str(i), PCT(nIn=C*4, d=dilation_block_2[i], p=P))
        
        # Decoder
        self.Decoder = SimpleDecoder(C*2, C*4, num_classes)
    
    def forward(self, input):
        output0 = self.Init_Block(input)
        output1 = self.LC_Block_1(output0)
        output2 = self.LC_Block_2(output1)
        
        out = self.Decoder(output1, output2)
        
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        return out