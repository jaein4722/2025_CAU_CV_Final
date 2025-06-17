import torch
import torch.nn as nn
import torch.nn.functional as F

# (DSConv, DownSamplingBlock, Split, TCA, PCT, SEBlock 등 다른 모듈은 이전과 동일하게 유지)
# ... (이전 모듈 정의 코드) ...

class DSConv(nn.Module):
    """
    Depthwise Separable Convolution 모듈입니다.
    ★★★ 개선점 3 적용: 활성화 함수를 SiLU로 통일 ★★★
    """
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=1, bn_acti=True):
        super().__init__()
        self.bn_acti = bn_acti
        # Atrous Convolution을 지원하기 위해 dilation 파라미터 추가
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding, dilation=dilation, groups=nIn, bias=False)
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)
        
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True) # SELU -> SiLU 로 변경

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
        
        # DSConv를 사용하도록 변경 (경량화 및 일관성)
        self.conv3x3 = DSConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output_conv = self.conv3x3(input)
        
        max_pool = self.max_pool(input)
        # 채널 수가 부족할 때만 concat하도록 수정 (nOut > nIn)
        if self.nIn < self.nOut:
            output = torch.cat([output_conv, max_pool], 1)
        else: # 채널 수가 같다면 conv 결과만 사용
            output = output_conv

        output = self.bn(output)
        output = self.acti(output)
        return output

def Split(x, p):
    c = int(x.size(1))
    c1 = round(c * (1-p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class TCA(nn.Module):
    def __init__(self, c, d=1, kSize=3):
        super().__init__()
        # DSConv에 dilation을 적용하기 위해 padding 계산 방식 변경
        padding = d * (kSize - 1) // 2
        self.conv3x3 = DSConv(c, c, kSize, 1, padding=padding, dilation=d, bn_acti=True)

    def forward(self, input):
        output = self.conv3x3(input)
        return output

class PCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.TCA = TCA(c, d)
        self.conv1x1 = nn.Conv2d(nIn, nIn, 1, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        output = self.bn(output)
        output = self.acti(output)
        return output + input  # 잔차 연결 추가

class SEBlock(nn.Module):
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

# --- ★★★★★ 여기가 수정된 부분 ★★★★★ ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = DSConv(in_channels, out_channels, kSize=1, stride=1, padding=0, dilation=dilations[0], bn_acti=True)
        self.aspp2 = DSConv(in_channels, out_channels, kSize=3, stride=1, padding=dilations[1], dilation=dilations[1], bn_acti=True)
        self.aspp3 = DSConv(in_channels, out_channels, kSize=3, stride=1, padding=dilations[2], dilation=dilations[2], bn_acti=True)
        self.aspp4 = DSConv(in_channels, out_channels, kSize=3, stride=1, padding=dilations[3], dilation=dilations[3], bn_acti=True)
        
        # --- 수정된 global_avg_pool ---
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels), # <<< 오류를 유발하는 이 줄을 삭제합니다.
            nn.SiLU(inplace=True)
        )
            
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        out = self.conv1(x_cat)
        out = self.bn1(out)
        out = self.relu(out)
        
        return self.dropout(out)

class FusionBlock(nn.Module):
    def __init__(self, c_high, c_low, c_out):
        super().__init__()
        self.conv_low = nn.Conv2d(c_low, c_high, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(c_high)
        self.fusion_conv = DSConv(c_high * 2, c_out, kSize=3, stride=1, padding=1, bn_acti=True)
    def forward(self, x_high, x_low):
        x_low_upsampled = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_low_processed = self.bn_low(self.conv_low(x_low_upsampled))
        fused_features = torch.cat([x_high, x_low_processed], dim=1)
        out = self.fusion_conv(fused_features)
        return out

class MultiStageDecoder(nn.Module):
    def __init__(self, c_high, c_mid, c_low, num_classes):
        super().__init__()
        self.up_1 = FusionBlock(c_mid, c_low, c_mid)
        self.up_2 = FusionBlock(c_high, c_mid, c_high)
        self.output_conv = nn.Conv2d(c_high, num_classes, kernel_size=1, bias=False)
    def forward(self, x_high, x_mid, x_low):
        fused_mid = self.up_1(x_mid, x_low)
        fused_high = self.up_2(x_high, fused_mid)
        out = self.output_conv(fused_high)
        return out

# --------------------------------------------------------------------------------
# 최종 제출용 모델 (모든 개선안 통합)
# --------------------------------------------------------------------------------
class submission_LCNetv6(nn.Module): # ★★★ 최종 제출 시 submission_{본인학번}으로 변경 필수 ★★★
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        block_1 = 2
        block_2 = 6
        C = 8
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        se_reduction = 4
        self.Init_Block = nn.Sequential(
            DSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C*2))
        for i in range(block_1):       
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(nIn=C*2, d=dilation_block_1[i], p=P))
        self.LC_Block_1.add_module("SE_Block", SEBlock(channel=C*2, reduction=se_reduction))
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C*2, C*4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(nIn=C*4, d=dilation_block_2[i], p=P))
        self.LC_Block_2.add_module("SE_Block", SEBlock(channel=C*4, reduction=se_reduction))
        self.aspp = ASPP(in_channels=C*4, out_channels=C*4)
        self.Decoder = MultiStageDecoder(C, C*2, C*4, num_classes)
    
    def forward(self, input):
        output0 = self.Init_Block(input)
        output1 = self.LC_Block_1(output0)
        output2 = self.LC_Block_2(output1)
        output2_aspp = self.aspp(output2)
        out = self.Decoder(output0, output1, output2_aspp)
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)
        return out
    
if __name__ == "__main__":
    net = submission_LCNetv6(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # 파라미터가 아주 미세하게 감소합니다 (BatchNorm 레이어의 가중치, 편향 제거)
    print(f"Trainable Params: {p/1e3:.1f} K")