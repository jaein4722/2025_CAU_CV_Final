import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
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

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1, bn_acti=False)
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
        self.conv3x3 = Conv(c, c, kSize, 1, padding=1, bn_acti=True)
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

class PCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.TCA = TCA(c, d)
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output + input

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            Conv(in_channels // 2 + skip_channels, out_channels, 3, 1, padding=1, bn_acti=True),
            Conv(out_channels, out_channels, 3, 1, padding=1, bn_acti=True)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class MorphGradientFocus(nn.Module):
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

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W) 
        k = self.key(x).view(B, -1, H * W)   
        v = self.value(x).view(B, -1, H * W) 

        attn = torch.bmm(q.permute(0, 2, 1), k) 
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)) 
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class submission_HWNetv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        block_1 = 4
        block_2 = 8
        C = 16
        P = 0.3
        dilation_block_1 = [2, 2, 2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16, 32, 32]

        self.edge_focus = MorphGradientFocus(in_channels, k=3)

        self.Init_Block = nn.Sequential(
            Conv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True)
        )

        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module(f"LC_Module_1_{i}", PCT(C * 2, d=dilation_block_1[i], p=P))

        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module(f"LC_Module_2_{i}", PCT(C * 4, d=dilation_block_2[i], p=P))

        self.attn = SpatialSelfAttention(C * 4)

        self.dec2 = DecoderBlock(C * 4, C * 2, C * 2)
        self.dec1 = DecoderBlock(C * 2, C, C)
        self.final_up = nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(C // 2, num_classes, kernel_size=1)

    def forward(self, input):
        input = self.edge_focus(input)
        x0 = self.Init_Block(input)
        x1 = self.LC_Block_1(x0)
        x2 = self.LC_Block_2(x1)
        x2 = self.attn(x2)
        d2 = self.dec2(x2, x1)
        d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        out = self.final_conv(out)
        return out

if __name__ == "__main__":
    net = submission_HWNetv3(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    
    
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # 파라미터가 아주 미세하게 감소합니다 (BatchNorm 레이어의 가중치, 편향 제거)
    print(f"Trainable Params: {p/1e3:.1f} K")