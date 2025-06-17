import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pointwise(out)
        return out

class MultiDilationSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2 
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.depthwise2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        out = x1 + x2
        out = self.pointwise(out)
        return out

class MicroDownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels
        channels_conv = out_channels if not self.use_maxpool else out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)
        out = self.bn(out)
        return F.relu(out)

class MicroResidualConvModule(nn.Module):
    def __init__(self, channels, dilation, dropout=0):
        super().__init__()
        self.conv = SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class MicroResidualMultiDilationConvModule(nn.Module):
    def __init__(self, channels, dilation, dropout=0):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.dropout(out)
        return F.relu(x + out)

class MicroUpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)

class GradientFeatureModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.refine_conv = SeparableConv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.relu(out)

class MicroNetV5Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample_1 = MicroDownsampleModule(in_channels, 10)
        self.gradient_feature_module = GradientFeatureModule(10)
        self.downsample_2 = MicroDownsampleModule(10, 20)
        self.downsample_modules = nn.Sequential(
            MicroResidualConvModule(20, 1, 0),
            MicroResidualConvModule(20, 1, 0)
        )
        self.downsample_3 = MicroDownsampleModule(20, 26)
        rates = [1, 2, 4, 8] 
        self.feature_modules = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(26, rate, 0.1) for rate in rates
        ])

    def forward(self, x):
        d1 = self.downsample_1(x)
        g_feat = self.gradient_feature_module(d1)
        d1_enhanced = d1 + g_feat 
        d2 = self.downsample_2(d1_enhanced)
        m2 = self.downsample_modules(d2)
        d3 = self.downsample_3(m2)
        m4 = self.feature_modules(d3)
        return m4, d2

class submission_MiniNetV3(nn.Module):
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        self.encoder = MicroNetV5Encoder(in_channels)
        self.aux_downsample = MicroDownsampleModule(in_channels, 10)
        self.aux_refine = MicroResidualConvModule(10, 1, 0)
        self.upsample_1 = MicroUpsampleModule(26, 20)
        self.upsample_mods = nn.Sequential(
            MicroResidualConvModule(20, 1, 0),
            MicroResidualConvModule(20, 1, 0)
        )
        self.output_conv = nn.ConvTranspose2d(20, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        aux = self.aux_downsample(x)
        aux = self.aux_refine(aux)
        enc, skip = self.encoder(x)
        up1 = self.upsample_1(enc)
        if up1.shape[2:] == skip.shape[2:]:
            up1 = up1 + skip
        if up1.shape[2:] == aux.shape[2:]:
            up1 = up1 + aux
        m3 = self.upsample_mods(up1)
        out = self.output_conv(m3)
        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

if __name__ == "__main__":
    net = submission_MiniNetV3(in_channels=3, num_classes=2)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters())
    print(f"Params: {p/1e3:.1f} K")