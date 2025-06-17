import torch
import torch.nn as nn
import torch.nn.functional as F

# [전략 2] 표준 Conv를 대체할 깊이별 분리형 컨볼루션(DWS-Conv) 클래스
class DWSConv(nn.Module):
    """Depthwise Separable Convolution. 파라미터를 크게 줄이는 핵심 모듈."""
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, dilation=(1, 1), groups=1, bn_acti=True, bias=False):
        super().__init__()
        # groups=nIn -> Depthwise Convolution
        self.depthwise = nn.Conv2d(nIn, nIn, kernel_size=kSize, stride=stride, padding=padding,
                                   dilation=dilation, groups=nIn, bias=bias)
        # 1x1 Conv -> Pointwise Convolution
        self.pointwise = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn_acti = bn_acti
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.acti = nn.SiLU(inplace=True) # 경량 모델에서 좋은 성능을 보이는 SiLU(Swish) 활성화 함수 사용

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_acti:
            x = self.bn(x)
            x = self.acti(x)
        return x

class Conv(nn.Module):
    """Pointwise(1x1) Convolution 또는 일반 Conv를 위한 기존 클래스"""
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
    """DWS-Conv를 사용하도록 수정한 다운샘플링 블록"""
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        # [전략 2] 파라미터 절감을 위해 DWSConv 사용
        self.conv3x3 = DWSConv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
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

class TCA(nn.Module):
    def __init__(self, c, d=1, kSize=3, dkSize=3):
        super().__init__()
        # [전략 2] 3x3 Conv를 DWSConv로 교체
        self.conv3x3 = DWSConv(c, c, kSize, 1, padding=1, bn_acti=True)
        # Depthwise Conv는 원래 파라미터가 매우 적으므로 그대로 유지
        self.dconv3x3 = Conv(c, c, (dkSize, dkSize), 1, padding=(1, 1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c, c, (dkSize, dkSize), 1, padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)
        self.bn = nn.BatchNorm2d(c, eps=1e-3)
        self.acti = nn.SiLU(inplace=True)

    def forward(self, input):
        br = self.conv3x3(input)
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        output = self.bn(br)
        output = self.acti(output)
        return output

class PCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.TCA = TCA(c, d)
        # 1x1 Conv는 원래 효율적이므로 기존 Conv 사용
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output + input

class DecoderBlock(nn.Module):
    """DWS-Conv를 사용하도록 수정한 디코더 블록"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # [전략 2] 모든 3x3 Conv를 DWSConv로 교체
        self.conv = nn.Sequential(
            DWSConv(in_channels // 2 + skip_channels, out_channels, 3, 1, padding=1, bn_acti=True),
            DWSConv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class MorphGradientFocus(nn.Module):
    """파라미터가 거의 없는 효율적인 경계 강화 모듈 (그대로 유지)"""
    def __init__(self, in_channels, k: int = 3):
        super().__init__()
        self.pad  = k // 2
        self.fuse = Conv(in_channels + 1, in_channels, 1, 1, padding=0, bn_acti=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intensity = x.mean(dim=1, keepdim=True)
        dilated = F.max_pool2d(intensity, 3, stride=1, padding=self.pad)
        eroded  = -F.max_pool2d(-intensity, 3, stride=1, padding=self.pad)
        edge    = dilated - eroded
        out = self.fuse(torch.cat([x, edge], dim=1))
        return out

# --- 최종 제출 모델 ---
class submission_HWNetNano_v1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # [전략 1] 모델의 너비와 깊이를 대폭 축소
        block_1 = 2
        block_2 = 2
        C = 12
        P = 0.5
        dilation_block_1 = [2, 2]
        dilation_block_2 = [4, 4]

        # [전략 4] 파라미터 효율적인 모듈은 유지
        self.edge_focus = MorphGradientFocus(in_channels, k=3)

        # 초기 블록도 DWSConv로 경량화
        self.Init_Block = nn.Sequential(
            DWSConv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            DWSConv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))

        # [전략 3] 고비용 어텐션 모듈 제거
        # self.attn = SpatialSelfAttention(C * 4)

        self.dec2 = DecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = DecoderBlock(C * 2, C, C)
        self.final_up = nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(C // 2, num_classes, kernel_size=1)

    def forward(self, input):
        input = self.edge_focus(input)
        x0 = self.Init_Block(input)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        # 어텐션 모듈 제거로 forward 경로 수정
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        out = self.final_conv(out)
        return out

if __name__ == "__main__":
    # 프로젝트의 Model_Test.ipynb와 동일한 환경으로 테스트
    num_classes = 21 # Pascal VOC와 같은 다중 클래스 케이스
    net = submission_HWNetNano_v1(in_channels=3, num_classes=num_classes)
    
    # 모델 구조 및 파라미터 수 확인
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: HWNet-Nano_v1")
    print(f"Trainable Params: {p/1e3:.2f} K") # 목표: < 10 K

    # 더미 입력으로 실행 테스트
    try:
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("Test Passed: Model runs successfully and output shape is correct.")
    except Exception as e:
        print(f"Test Failed: {e}")