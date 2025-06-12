import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial Squeeze Block (S2-block): 입력 해상도를 유지하며 풀링 후 depthwise conv 수행
class S2Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, kernel_size, padding=kernel_size//2)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 원본 해상도 보존을 위해 F.interpolate 사용
        size = x.shape[2:]
        out = self.pool(x)
        out = self.dwconv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out

# 채널 셔플: group conv 후 채널 간 교환을 수행
class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batchsize, -1, height, width)

# SIUnit: Spatial Squeeze Module + Residual 연결
class SIUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.reduce = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid)
        self.shuffle = ChannelShuffle(groups=2)
        self.s2b1 = S2Block(mid, kernel_size=3)
        self.s2b2 = S2Block(mid, kernel_size=5)
        self.merge = nn.Conv2d(mid * 2, channels, kernel_size=1, bias=False)
        self.bn_merge = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        identity = x
        out = self.reduce(x)
        out = self.bn_reduce(out)
        out = self.shuffle(out)
        b1 = self.s2b1(out)
        b2 = self.s2b2(out)
        out = torch.cat([b1, b2], dim=1)
        out = self.merge(out)
        out = self.bn_merge(out)
        return self.prelu(out + identity)

# SINet 메인 클래스
class submission_SINet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        width_mult = 0.5
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        self.SI = SIUnit(channels=c3)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.encoder3(x)
        x = self.SI(x)
        x = self.up1(x)
        x = self.decoder1(x)
        x = self.up2(x)
        x = self.decoder2(x)
        return self.classifier(x)