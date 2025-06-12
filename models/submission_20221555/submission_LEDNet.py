
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate

def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output

class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        oup_inc = chann // 2
        
        # dw left
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        # dw right
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)
        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated))
        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
        
        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1, output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 32)
        self.layers = nn.ModuleList()
        
        for _ in range(3): self.layers.append(SS_nbt_module(32, 0.03, 1))
        self.layers.append(DownsamplerBlock(32, 64))
        for _ in range(2): self.layers.append(SS_nbt_module(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        
        self.layers.append(SS_nbt_module(128, 0.3, 1))
        self.layers.append(SS_nbt_module(128, 0.3, 2))
        self.layers.append(SS_nbt_module(128, 0.3, 5))
        self.layers.append(SS_nbt_module(128, 0.3, 9))
        self.layers.append(SS_nbt_module(128, 0.3, 2))
        self.layers.append(SS_nbt_module(128, 0.3, 5))
        self.layers.append(SS_nbt_module(128, 0.3, 9))
        self.layers.append(SS_nbt_module(128, 0.3, 17))

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        return output

class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0))
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)
        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)
        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
        )
        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        
        b1 = self.branch1(x)
        b1 = interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)
        
        mid = self.mid(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x3 = interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)
        x2 = self.conv2(x2)
        x_attn = x2 + x3
        x_attn = interpolate(x_attn, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        
        x1 = self.conv1(x1)
        x_attn = x_attn + x1
        x_attn = interpolate(x_attn, size=(h, w), mode="bilinear", align_corners=True)
        
        x = torch.mul(x_attn, mid)
        x = x + b1
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.apn = APN_Module(in_ch=128, out_ch=num_classes)
        # 프로젝트에서 입력 이미지는 256x256 이므로 출력도 동일하게 맞춰줍니다.
        # 원본의 upsample 크기는 Cityscapes 데이터셋(512x1024)에 맞춰져 있었습니다.
        self.upsample_size = (256, 256)

    def forward(self, input):
        output = self.apn(input)
        # 최종 출력을 입력과 동일한 256x256 크기로 보간합니다.
        out = interpolate(output, size=self.upsample_size, mode="bilinear", align_corners=True)
        return out

class submission_LEDNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        # 프로젝트 가이드라인에 따라 in_channels와 num_classes를 받습니다.
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, input):
        output = self.encoder(input)
        return self.decoder.forward(output)