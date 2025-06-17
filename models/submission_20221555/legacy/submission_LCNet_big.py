import torch
import torch.nn as nn
import torch.nn.functional as F

# [개선점 1] 일반 Conv를 대체할 DSConv 모듈 정의
class DSConv(nn.Module):
    """Depthwise Separable Convolution. SELU 활성화 함수 사용."""
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), bias=False):
        super().__init__()
        # kSize=1인 경우, 일반 Conv와 동일하게 동작 (Pointwise)
        if kSize == 1:
            self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=bias)
        else:
            self.conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding,
                          dilation=dilation, groups=nIn, bias=bias),
                # Pointwise
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=bias)
            )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.acti = nn.SELU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.acti(output)
        return output

# --- 기존 모듈들은 그대로 사용 ---
class Conv(nn.Module): # 1x1 Conv 나 depthwise conv를 위해 유지
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SELU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn(output)
            output = self.acti(output)
        return output

class DownSamplingBlock_DS(nn.Module): # DSConv를 사용하도록 수정
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        # [개선점 1] 3x3 Conv를 DSConv로 교체
        self.conv3x3 = DSConv(nIn, nConv, kSize=3, stride=2, padding=1)
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
    c1 = round(c * (1 - p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class TCA_DS(nn.Module): # DSConv를 사용하도록 수정
    def __init__(self, c, d=1, kSize=3, dkSize=3):
        super().__init__()
        # [개선점 1] 3x3 Conv를 DSConv로 교체
        self.conv3x3 = DSConv(c, c, kSize, 1, padding=1)
        # 아래 두 conv는 이미 depthwise이므로 교체 불필요
        self.dconv3x3 = Conv(c, c, (dkSize, dkSize), 1, padding=(1, 1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c, c, (dkSize, dkSize), 1, padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)
        self.bn = nn.BatchNorm2d(c, eps=1e-3)
        self.acti = nn.SELU(inplace=True)

    def forward(self, input):
        br = self.conv3x3(input)
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        output = self.bn(br)
        output = self.acti(output)
        return output

class PCT_DS(nn.Module): # TCA_DS를 사용하도록 수정
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.TCA = TCA_DS(c, d)
        # 1x1 Conv는 효율적이므로 교체 불필요
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output + input # 이미 residual connection이 있음

class DecoderBlock_DS(nn.Module): # DSConv를 사용하도록 수정
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            # [개선점 1] 3x3 Conv들을 DSConv로 교체
            DSConv(in_channels // 2 + skip_channels, out_channels, 3, 1, padding=1),
            DSConv(out_channels, out_channels, 3, 1, padding=1)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

# --- 최종 모델 ---
class submission_LCNet_big(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        block_1 = 4
        block_2 = 8
        C = 16
        P = 0.3
        dilation_block_1 = [2, 2, 2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16, 32, 32]

        # [개선점 1] Init_Block의 3x3 Conv들을 DSConv로 교체
        self.Init_Block = nn.Sequential(
            DSConv(in_channels, C, 3, 2, padding=1),
            DSConv(C, C, 3, 1, padding=1),
            DSConv(C, C, 3, 1, padding=1)
        )
        
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock_DS(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT_DS(C * 2, d=dilation_block_1[i], p=P))

        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock_DS(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT_DS(C * 4, d=dilation_block_2[i], p=P))
            
        # [개선점 2] LC_Block을 위한 잔차 연결용 프로젝션 레이어 추가
        self.proj1 = Conv(C, C * 2, kSize=1, stride=2, padding=0, bn_acti=True)
        self.proj2 = Conv(C * 2, C * 4, kSize=1, stride=2, padding=0, bn_acti=True)

        # [개선점 1] DecoderBlock을 DSConv 버전으로 교체
        self.dec2 = DecoderBlock_DS(C * 4, C * 2, C * 2)
        self.dec1 = DecoderBlock_DS(C * 2, C, C)
        
        self.final_up = nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2)
        # 마지막 1x1 Conv는 교체 불필요
        self.final_conv = nn.Conv2d(C // 2, num_classes, kernel_size=1)

    def forward(self, input):
        x0 = self.Init_Block(input)
        
        # [개선점 2] LC_Block_1에 잔차 연결 적용
        x1_res = self.proj1(x0) # 차원 맞추기
        x1 = self.LC_Block_1(x0) + x1_res
        
        # [개선점 2] LC_Block_2에 잔차 연결 적용
        x2_res = self.proj2(x1) # 차원 맞추기
        x2 = self.LC_Block_2(x1) + x2_res
        
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        out = self.final_conv(out)
        return out
    
# -------------------------------------------------
# 3. Quick Test
# -------------------------------------------------
if __name__ == "__main__":
    net = submission_LCNet_big(in_channels=3, num_classes=21)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters())
    print(f"Params: {p/1e3:.1f} K")