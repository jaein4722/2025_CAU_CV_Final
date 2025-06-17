import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# 1. Core Blocks
# -------------------------------------------------
class DSConv(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 use_residual: bool = True,
                 dilation: int | tuple[int, int] = 1):
        super().__init__()
        self.use_residual = use_residual

        # ---- ① dilation 정규화 ----
        if isinstance(dilation, bool):          # True / False → 1
            dilation = 1
        if isinstance(dilation, int):
            pad = dilation
        elif (isinstance(dilation, tuple) and
              all(isinstance(d, int) for d in dilation)):
            pad = dilation
        else:                                   # 그 외 타입 방지
            raise ValueError("dilation must be int or tuple of ints")

        # ---- ② Depthwise + Pointwise ----
        self.depthwise = nn.Conv2d(
            c_in, c_in,
            kernel_size=3,
            padding=pad,
            dilation=dilation,
            groups=c_in,
            bias=False
        )
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

        self.res_proj = (
            nn.Conv2d(c_in, c_out, 1, bias=False)
            if (use_residual and c_in != c_out) else nn.Identity()
        )

    def forward(self, x):
        identity = self.res_proj(x)
        out = self.bn(self.pointwise(self.depthwise(x)))
        if self.use_residual:
            out += identity
        return self.act(out)


class InvertedResidual(nn.Module):
    """MobileNetV2 style: expand → DW → project"""
    def __init__(self, c, expand_ratio=2):
        super().__init__()
        mid = c * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )
    def forward(self, x):
        return x + self.conv(x)

class CoordAtt(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        m = max(8, c // r)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c, m, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(m, c, 1, bias=False)
        self.conv_w = nn.Conv2d(m, c, 1, bias=False)
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv1(y))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w).permute(0,1,3,2))
        return x * a_h * a_w

class LiteASPP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        rates = [1, 4, 8, 12]
        self.branches = nn.ModuleList([
            DSConv(c_in, c_out, dilation=r, use_residual=False) for r in rates
        ] + [nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_in, c_out, 1, bias=False), nn.ReLU(inplace=True))])
        self.project = nn.Conv2d(c_out*5, c_out, 1, bias=False)
    def forward(self, x):
        h, w = x.shape[2:]
        outs = []
        for b in self.branches:
            y = b(x)
            if y.shape[2:] != (h, w):
                y = F.interpolate(y, (h, w), mode='nearest')
            outs.append(y)
        return self.project(torch.cat(outs, 1))

# -------------------------------------------------
# 2. Micro‑UNet v3
# -------------------------------------------------
class submission_MicroUNetv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, use_residual=True, use_aux=False):
        super().__init__()
        self.use_residual, self.use_aux = use_residual, use_aux
        b = 12  # base channels

        # -------- Encoder --------
        self.enc1 = nn.Sequential(DSConv(in_channels, b, use_residual), DSConv(b, b, use_residual))
        self.enc2 = nn.Sequential(DSConv(b, b*2, use_residual), DSConv(b*2, b*2, use_residual))
        self.enc3 = nn.Sequential(DSConv(b*2, b*4, use_residual), DSConv(b*4, b*4, use_residual))
        self.pool = nn.MaxPool2d(2)

        # -------- Context --------
        self.ctx = nn.Sequential(
            DSConv(b*4, b*4, dilation=2, use_residual=False),
            DSConv(b*4, b*4, dilation=4, use_residual=False),
            InvertedResidual(b*4, expand_ratio=2),
            LiteASPP(b*4, b*4)
        )
        self.att_e3 = CoordAtt(b*4)

        # -------- Decoder --------
        # up1: 48→24
        self.up1_pre = DSConv(b*4, b*2, use_residual)
        self.fuse2 = nn.Conv2d(b*4, b*2, 1, bias=False)
        self.att_d2 = CoordAtt(b*2)
        # up2: 24→12
        self.up2_pre = DSConv(b*2, b, use_residual)
        self.fuse1 = nn.Conv2d(b*2, b, 1, bias=False)

        # -------- Heads --------
        self.final_conv = nn.Conv2d(b, num_classes, 1)
        self.br         = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        if use_aux:
            self.aux_head = nn.Conv2d(b*2, num_classes, 1)

        self._init_weights()

    # -------------------------------------------------
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # B,12,H,W
        p1 = self.pool(e1)
        e2 = self.enc2(p1)         # B,24,H/2,W/2
        p2 = self.pool(e2)
        e3 = self.enc3(p2)         # B,48,H/4,W/4
        e3 = self.att_e3(e3)

        # Context
        b = self.ctx(e3)
        if self.use_residual: b = b + e3

        # Decoder stage 1 (H/4→H/2)
        d2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up1_pre(d2)
        d2 = torch.cat([d2, e2], 1)  # 24 + 24
        d2 = self.fuse2(d2)          # →24
        d2 = self.att_d2(d2)

        # Decoder stage 2 (H/2→H)
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up2_pre(d1)
        d1 = torch.cat([d1, e1], 1)  # 12+12
        d1 = self.fuse1(d1)          # →12

        # Heads
        logits = self.final_conv(d1)
        #logits = logits + self.br(logits)
        if self.use_aux:
            aux = self.aux_head(d2)
            aux = F.interpolate(aux, size=x.shape[2:], mode='bilinear', align_corners=False)
            return logits, aux
        return logits

    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

# -------------------------------------------------
# 3. Quick Test
# -------------------------------------------------
if __name__ == "__main__":
    model = submission_MicroUNetv3(in_channels=3, num_classes=21, use_residual=True, use_aux=False)
    x = torch.randn(1,3,256,256)
    y = model(x)
    if isinstance(y, tuple):
        print("main", y[0].shape, "aux", y[1].shape)
    else:
        print("out", y.shape)
    p = sum(p.numel() for p in model.parameters())
    print(f"Params: {p/1e3:.1f} K")