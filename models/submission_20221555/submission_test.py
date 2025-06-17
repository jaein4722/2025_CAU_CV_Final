import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        y = F.relu(self.bn(self.depthwise(x)))
        return self.pointwise(y)

class MultiDilationSeparableConv2d(nn.Module):
    def __init__(self, ch, out_ch, k=3, stride=1, pad=0, dil=1, bias=True):
        super().__init__()
        pad2 = pad + (dil - 1) * (k - 1) // 2
        self.d1 = nn.Conv2d(ch, ch, k, stride, pad, 1, groups=ch, bias=False)
        self.d2 = nn.Conv2d(ch, ch, k, stride, pad2, dil, groups=ch, bias=False)
        self.b1 = nn.BatchNorm2d(ch, eps=1e-3)
        self.b2 = nn.BatchNorm2d(ch, eps=1e-3)
        self.pw = nn.Conv2d(ch, out_ch, 1, bias=bias)

    def forward(self, x):
        a = F.relu(self.b1(self.d1(x)))
        b = F.relu(self.b2(self.d2(x)))
        return self.pw(a + b)

class DownsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_pool = in_ch < out_ch
        conv_ch = out_ch if not self.use_pool else out_ch - in_ch
        self.conv = nn.Conv2d(in_ch, conv_ch, 3, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2)], 1)
        return F.relu(self.bn(y))

class ResidualConvModule(nn.Module):
    def __init__(self, ch, dil, drop=0.):
        super().__init__()
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, bias=False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.do   = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.do(self.bn(self.conv(x)))
        return F.relu(x + y)

class ResidualMultiDilationConvModule(nn.Module):
    def __init__(self, ch, dil, drop=0.):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, 1, 1, dil, bias=False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.do   = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.do(self.bn(self.conv(x)))
        return F.relu(x + y)

class UpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ECA(nn.Module):

    def __init__(self, ch, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).view(x.size(0), 1, -1)       
        y = self.conv(y)
        y = self.sigm(y).view(x.size(0), -1, 1, 1)   
        return x * y

class LiteASPP(nn.Module):
    def __init__(self, in_ch, mid_ch=5): 
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_branch = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b1 = SeparableConv2d(in_ch, mid_ch, 3, 1, 1, 1, bias=False)
        self.b2 = SeparableConv2d(in_ch, mid_ch, 3, 1, 2, 2, bias=False)
        self.b3 = SeparableConv2d(in_ch, mid_ch, 3, 1, 4, 4, bias=False)
        self.project = nn.Conv2d(mid_ch * 4, in_ch, 1, bias=False)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        p = F.interpolate(self.pool_branch(self.pool(x)), size=(h, w), mode='bilinear', align_corners=False)
        y = torch.cat([p, self.b1(x), self.b2(x), self.b3(x)], 1)
        return F.relu(self.project(y))

class MiniNetV2Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.d1 = DownsampleModule(in_ch, 8)
        self.d2 = DownsampleModule(8, 16)
        self.res = nn.Sequential(*[ResidualConvModule(16, 1) for _ in range(3)])
        self.d3 = DownsampleModule(16, 32)
        self.feat = nn.Sequential(*[ResidualMultiDilationConvModule(32, r, 0.1) for r in (1, 2, 4)])

    def forward(self, x):
        s1 = self.d1(x)         
        s2 = self.d2(s1)        
        s2 = self.res(s2)       
        s3 = self.d3(s2)        
        return self.feat(s3), s2

class MorphGradientFocus(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.pad = k // 2
        self.fuse = nn.Sequential(
            nn.Conv2d(ch + 1, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        g = x.mean(1, keepdim=True)
        edge = F.max_pool2d(g, 3, 1, self.pad) - (-F.max_pool2d(-g, 3, 1, self.pad))
        return self.fuse(torch.cat([x, edge], 1))

class submission_test(nn.Module):
    def __init__(self, in_ch, n_cls, interpolate=True):
        super().__init__()
        self.interpolate = interpolate

        self.encoder = MiniNetV2Encoder(in_ch)
        self.aspp    = LiteASPP(32)
        self.eca     = ECA(32)           

        self.aux1 = DownsampleModule(in_ch, 8)
        self.aux2 = DownsampleModule(8, 16)
        self.mgf  = MorphGradientFocus(16)

        self.up1  = UpsampleModule(32, 16)
        self.fuse = nn.Sequential(
            nn.Conv2d(48, 16, 1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.up_blocks = nn.Sequential(*[ResidualConvModule(16, 1) for _ in range(3)]) 
        self.head = nn.ConvTranspose2d(16, n_cls, 3, 2, 1, 1, bias=True)

    def forward(self, x):
        enc, skip = self.encoder(x)
        enc = self.eca(self.aspp(enc))          

        aux = self.mgf(self.aux2(self.aux1(x))) 
        up  = self.up1(enc)                     

        y = self.fuse(torch.cat([up, aux, skip], 1))
        y = self.up_blocks(y)
        y = self.head(y)                        
        if self.interpolate:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
        return y