import torch
import torch.nn as nn
import torch.nn.functional as F

"""
모델 : submission_MicroUNet
· Depth‑wise Separable Conv + LiteASPP + 선택적 Residual / Skip‑connection
· 파라미터 ≈ 0.05M (채널 8‑16‑32)

함수 :
    ├─ DSConv          ‑ dilation 지원, residual ON/OFF
    ├─ LiteASPP        ‑ Mobile‑friendly ASPP (dil=1/6/12/18 + img‑pool)
    ├─ submission_MicroUNet
    ├─ TverskyLoss     ‑ α, β 가중치 지원, reduction="mean"
    ├─ FocalLoss       ‑ γ 지수, BCE 기반 focal
    └─ ComboLoss       ‑ λ*Tversky + (1‑λ)*Focal   (다중 클래스 OK)

사용 예시::

    model = submission_MicroUNet(in_channels=3,
                                 num_classes=21,
                                 use_residual=True)
    criterion = ComboLoss(alpha=0.7, beta=0.3, gamma=2.0, weight_tversky=0.6)
    optim = torch.optim.AdamW(model.parameters(), lr=3e‑4, weight_decay=1e‑4)
"""

# -------------------------------------------------
# 1. Depth‑wise Separable Convolution (dilation+residual)
# -------------------------------------------------
class DSConv(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 use_residual: bool = True,
                 dilation: int = 1):
        super().__init__()
        self.use_residual = use_residual

        self.depthwise = nn.Conv2d(c_in, c_in,
                                   kernel_size=3,
                                   padding=dilation,
                                   dilation=dilation,
                                   groups=c_in,
                                   bias=False)
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

# -------------------------------------------------
# 2. Lite‑ASPP (Mobile‑friendly ASPP)
# -------------------------------------------------
class LiteASPP(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        rates = [1, 6, 12, 18]  # 첫 번째는 dilation=1 (표준)

        branches = []
        # dilated depth‑wise branches
        for r in rates:
            branches.append(DSConv(c_in, c_out, use_residual=False, dilation=r))
        # image‑level pooling branch
        branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.ReLU(inplace=True)
        ))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Conv2d(c_out * len(branches), c_out, 1, bias=False)

    def forward(self, x):
        outs = []
        h, w = x.shape[2:]
        for idx, b in enumerate(self.branches):
            y = b(x)
            if y.shape[2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="nearest")
            outs.append(y)
        return self.project(torch.cat(outs, 1))

# -------------------------------------------------
# 3. Micro‑UNet 본체
# -------------------------------------------------
class submission_MicroUNetv2(nn.Module):
    def __init__(self, in_channels: int = 3,
                 num_classes: int = 1,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        
        features = 8
        self.features = features

        # Encoder (3→8→16→32)
        self.enc1 = nn.Sequential(
            DSConv(in_channels, features,  use_residual),
            DSConv(features, features,  use_residual)
        )
        self.enc2 = nn.Sequential(
            DSConv(features, features * 2, use_residual),
            DSConv(features * 2, features * 2, use_residual)
        )
        self.enc3 = nn.Sequential(
            DSConv(features * 2, features * 4, use_residual),
            DSConv(features * 4, features * 4, use_residual)
        )
        self.pool = nn.MaxPool2d(2)

        # Lite‑ASPP bottleneck
        self.aspp = LiteASPP(features * 4, features * 4)

        # Decoder (32→16→8)
        self.up1 = nn.Sequential(
            DSConv(features * 4, features * 2, use_residual),
            DSConv(features * 2, features * 2, use_residual),
        )
        self.up2 = nn.Sequential(
            DSConv(features * 2, features,  use_residual),
            DSConv(features, features,  use_residual)
        )

        # Boundary refinement (optional)
        self.br = DSConv(num_classes, num_classes, use_residual=False)

        self.head = nn.Conv2d(features, num_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        # -------- Bottleneck --------
        b = self.aspp(e3)
        if self.use_residual:
            b = b + e3

        # -------- Decoder --------
        d2 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up1(d2)
        if self.use_residual:
            d2 = d2 + e2

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up2(d1)
        if self.use_residual:
            d1 = d1 + e1

        # Head + Boundary refinement
        logits = self.head(d1)
        logits = logits + self.br(logits)
        return logits

    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)


# -------------------------------------------------
# 5. Quick Sanity Check
# -------------------------------------------------
if __name__ == "__main__":
    model = submission_MicroUNetv2(in_channels=3, num_classes=1, use_residual=True)
    x = torch.randn(2, 3, 256, 256)
    y = torch.randint(0, 2, (2, 1, 256, 256)).float()
    logits = model(x)
    print("logits", logits.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e3:.1f} K")
