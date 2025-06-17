import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic layers

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution layer.

    This layer:
    1. Performns a depthwise convolution, which convolves each input channel
       with a different kernel.
    2. Uses a standard 1x1 convolution (pointwise convolution) to combine the
       outputs of the depthwise convolution into the output feature maps.
    """
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
    """Multi-dilation depthwise separable convolution layer.

    This layer:
    1. Performs two parallel depthwise convolutions, one with dilation rate 1
       and another with dilation rate >= 1.
    2. Adds their outputs to combine features at different scales.
    3. Applies a pointwise convolution to produce the final feature maps.

    Note: 'kernel_size' and 'dilation' cannot both be even to ensure that both
    convolutions match in size.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,         1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
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


# MiniNet-v2 modules

class DownsampleModule(nn.Module):
    """Downsample module combining max pooling and a strided convolution.

    This approach helps in retaining resolution during the downsampling and is
    inpired by techniques used in models like DenseNet.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.use_maxpool = in_channels < out_channels

        if not self.use_maxpool:
            channels_conv = out_channels
        else:
            channels_conv = out_channels - in_channels

        self.conv = nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)

        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)

        out = self.bn(out)
        return F.relu(out)

class ResidualConvModule(nn.Module):
    """Residual convolution module using a separable convolution."""
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

class ResidualMultiDilationConvModule(nn.Module):
    """Residual convolution module using a multi-dilation separable convolution."""
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

class UpsampleModule(nn.Module):
    """Upsample module using a transposed convolution."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)


# MiniNet-v2 encoders

class MiniNetV2Encoder(nn.Module):
    """MiniNet-v2 encoder module."""
    def __init__(self, in_channels):
        super().__init__()

        # Downsample block
        self.downsample_1 = DownsampleModule(in_channels, 16)
        self.downsample_2 = DownsampleModule(16, 64)
        self.downsample_modules = nn.Sequential(*[ResidualConvModule(64, 1, 0) for _ in range(10)])
        self.downsample_3 = DownsampleModule(64, 128)

        # Feature extractor block
        rates = [1, 2, 1, 4, 1, 8, 1, 16, 1, 1, 1, 2, 1, 4, 1, 8]
        self.feature_modules = nn.Sequential(*[ResidualMultiDilationConvModule(128, rate, 0.25) for rate in rates])

    def forward(self, x):
        d1 = self.downsample_1(x)
        d2 = self.downsample_2(d1)
        m10 = self.downsample_modules(d2)
        d3 = self.downsample_3(m10)
        m25 = self.feature_modules(d3)
        return m25

class MiniNetV2EncoderCPU(MiniNetV2Encoder):
    """MiniNet-v2 encoder module optimized for CPU.

    This module is a smaller version of the standard MiniNet-v2 encoder that
    removes some of the convolutional modules to reduce the computational cost.
    """
    def __init__(self, in_channels):
        super().__init__(in_channels)

        self.downsample_modules = nn.Sequential(*[ResidualConvModule(64, 1, 0) for _ in range(2)])

        rates = [1, 2, 1, 4, 1, 8]
        self.feature_modules = nn.Sequential(*[ResidualMultiDilationConvModule(128, rate, 0.25) for rate in rates])


# MiniNet-v2 models for segmentation

class submission_MiniNetv2(nn.Module):
    """MiniNet-v2 model for semantic segmentation."""
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # Encoder
        self.encoder = MiniNetV2Encoder(in_channels)

        # Refinement block
        self.aux_downsample_4 = DownsampleModule(in_channels, 16)
        self.aux_downsample_5 = DownsampleModule(16, 64)

        # Upsample block
        self.upsample_1 = UpsampleModule(128, 64)
        self.upsample_mods = nn.Sequential(*[ResidualConvModule(64, 1, 0) for _ in range(4)])

        # Output
        self.output_conv = nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        aux = self.aux_downsample_4(x)
        aux = self.aux_downsample_5(aux)

        enc = self.encoder(x)
        up1 = self.upsample_1(enc)
        m29 = self.upsample_mods(up1 + aux)

        out = self.output_conv(m29)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == "__main__":
    # 테스트 및 파라미터 검증
    num_classes = 21
    net = submission_MiniNetv2(in_channels=3, num_classes=num_classes)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: MiniNetv2")
    print(f"Trainable Params: {p:,} ({p/1e3:.2f} K)")
    
    # 17k 제한 검증
    if p <= 10000:
        print(f"✅ 이상적 파라미터 범위: {p}/10,000")
    elif p <= 17000:
        print(f"⚠️  허용 범위 내: {p}/17,000 (hard cap)")
    else:
        print(f"❌ 파라미터 초과: {p}/17,000 ({p-17000} 초과)")

    try:
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == (1, num_classes, 256, 256)
        print("✅ 모델 실행 테스트 통과")
        
        # 다양한 클래스 수에 대한 테스트
        for nc in [1, 2, 21]:
            net_test = submission_MiniNetv2(in_channels=3, num_classes=nc)
            y_test = net_test(x)
            assert y_test.shape == (1, nc, 256, 256)
            print(f"✅ {nc} 클래스 테스트 통과")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}") 