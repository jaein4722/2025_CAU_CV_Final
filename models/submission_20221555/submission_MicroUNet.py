import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# 1. Depth-wise Separable Convolution (옵션 residual)
# -------------------------------------------------
class DSConv(nn.Module):
    """
    Depth-wise 3×3 (dilation 지원) → Point-wise 1×1 → BN → ReLU
    + optional residual
    """
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 use_residual: bool = True,
                 dilation: int = 1):        # ✅ dilation 추가
        super().__init__()
        self.use_residual = use_residual

        self.depthwise = nn.Conv2d(
            c_in, c_in,
            kernel_size=3,
            padding=dilation,              # ✅ same-size 유지
            dilation=dilation,             # ✅ dilation 적용
            groups=c_in,
            bias=False
        )
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

        if use_residual and c_in != c_out:
            self.res_proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        if self.use_residual:
            out = out + self.res_proj(x)
        return self.act(out)
    

class LiteASPP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        rates = [6, 12, 18]
        self.branches = nn.ModuleList([
            DSConv(c_in, c_out, use_residual=False),                       # 3×3 dil=1
            *[DSConv(c_in, c_out, use_residual=False, dilation=r)
              for r in rates],                                             # dilated
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                                   # Image pooling
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.ReLU(inplace=True)
            )
        ])
        self.project = nn.Conv2d(c_out*5, c_out, 1, bias=False)

    def forward(self, x):
        outs = [F.interpolate(b(x) if i>0 else self.branches[0](x),
                              size=x.shape[2:], mode="nearest")
                for i, b in enumerate(self.branches)]
        return self.project(torch.cat(outs, 1))


# -------------------------------------------------
# 2. Micro-UNet (Residual ON/OFF 스위치)
# -------------------------------------------------
class submission_MicroUNet(nn.Module):
    """
    Encoder : 3→8→16→32
    Bottleneck : GAP
    Decoder : 32→16→8→1
    use_residual = True  → 모든 skip 활성
    use_residual = False → 덧셈 skip 전부 비활성
    """
    def __init__(self, in_channels=3, num_classes=1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # ---------- Encoder ----------
        self.enc1 = DSConv(in_channels, 8,  use_residual)
        self.enc2 = DSConv(8,          16, use_residual)
        self.enc3 = DSConv(16,         32, use_residual)
        self.pool = nn.MaxPool2d(2)
        
        self.lite_aspp = LiteASPP(32, 32)

        # ---------- Decoder ----------
        self.up1 = DSConv(32, 16, use_residual)
        self.up2 = DSConv(16, 8,  use_residual)

        # ---------- Final ----------
        self.final_conv = nn.Conv2d(8, num_classes, kernel_size=1)

        self._init_weights()

    # -------------------------------------------------
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        # Bottleneck (GAP)
        b = self.lite_aspp(e3)
        if self.use_residual:
            b = b + e3

        # Decoder
        d2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up1(d2)               # 32 → 16
        if self.use_residual:
            d2 = d2 + e2                # 이제 16 vs 16 ➜ OK

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up2(d1)               # 16 → 8
        if self.use_residual:
            d1 = d1 + e1                # 8 vs 8 ➜ OK

        return self.final_conv(d1)

    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)


# -------------------------------------------------
# 3. 간단 테스트
# -------------------------------------------------
if __name__ == "__main__":
    for flag in [True, False]:
        model = submission_MicroUNet(use_residual=flag)
        params = sum(p.numel() for p in model.parameters())
        print(f"[Residual={flag}] params: {params/1e6:.3f} M")
        y = model(torch.randn(1, 3, 256, 256))
        print(" output shape:", y.shape)
