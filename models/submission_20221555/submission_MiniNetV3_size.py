"""
MiniNetV3 (파라미터화 버전)
────────────────────────────────────────────────────────
- 3 M 파라미터 이하 경량 세그멘테이션 네트워크
- (c1, c2, c3) 채널 튜플 하나만 바꿔서 폭 조정 가능
- 하이퍼파라미터(optimizer, lr, scheduler, loss)는 training_args.py에서 통제한다는
  프로젝트 규칙을 따르므로, 이 파일은 **모델 정의**만 포함한다.

※ 주석 및 문자열은 전부 한국어로 작성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────────────────────────────────────────────
# 👇 1. 기본 블록
# ───────────────────────────────────────────────────────────────
class SeparableConv2d(nn.Module):
    """3×3 Depthwise + 1×1 Pointwise 분리 합성곱"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)

    def forward(self, x):
        x = self.dw(x)
        x = F.relu(self.bn(x))
        x = self.pw(x)
        return x


class MultiDilationSeparableConv2d(nn.Module):
    """동일 입력을 dilation 1 / dilation d 로 나눠 처리 후 합산"""
    def __init__(self, in_ch, out_ch, k=3, d=2, bias=True):
        super().__init__()
        p1 = k // 2                        # dilation 1
        p2 = p1 + (d - 1) * (k - 1) // 2   # dilation d

        self.dw1 = nn.Conv2d(in_ch, in_ch, k, 1, p1,
                             groups=in_ch, bias=False)
        self.dw2 = nn.Conv2d(in_ch, in_ch, k, 1, p2, d,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_ch, eps=1e-3)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.dw1(x)))
        x2 = F.relu(self.bn2(self.dw2(x)))
        return self.pw(x1 + x2)


class MicroDownsampleModule(nn.Module):
    """
    stride-2 다운샘플 + (선택) MaxPool skip-cat
    - in_ch < out_ch → Conv(out_ch - in_ch) + MaxPool(in_ch) → concat
    - in_ch ≥ out_ch → Conv(out_ch) 단독
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_pool = in_ch < out_ch
        conv_out = out_ch if not self.use_pool else out_ch - in_ch

        self.conv = nn.Conv2d(in_ch, conv_out, 3, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2)], dim=1)
        return F.relu(self.bn(y))


class MicroResidualConvModule(nn.Module):
    """Separable Conv + Residual"""
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        return F.relu(x + y)


class MicroResidualMultiDilationConvModule(nn.Module):
    """Multi-Dilation Separable Conv + Residual"""
    def __init__(self, ch, dil=2, drop=0.):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        return F.relu(x + y)


class MicroUpsampleModule(nn.Module):
    """stride-2 Transposed Conv 업샘플"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1,
                                         output_padding=1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)))

# ───────────────────────────────────────────────────────────────
# 👇 2. Gradient Feature Module
# ───────────────────────────────────────────────────────────────
class GradientFeatureModule(nn.Module):
    """Sobel 필터로 채널별 x,y gradient 추출 후 1×1 리파인"""
    def __init__(self, ch):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.refine = SeparableConv2d(ch * 2, ch, 1, 1, 0, 1, False)
        self.bn     = nn.BatchNorm2d(ch, eps=1e-3)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        g  = torch.cat([gx, gy], dim=1)
        return F.relu(self.bn(self.refine(g)))

# ───────────────────────────────────────────────────────────────
# 👇 3. Encoder
# ───────────────────────────────────────────────────────────────
class MicroNetV5Encoder(nn.Module):
    """
    MicroNet-V5 기반 인코더
      ch = (c1, c2, c3)  : 다운샘플 단계별 채널 수
      rates              : Dilated 블록의 dilation 리스트
    """
    def __init__(self,
                 in_ch: int,
                 ch: tuple = (10, 20, 26),
                 rates: tuple = (1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3 = ch

        self.down1 = MicroDownsampleModule(in_ch, c1)
        self.grad  = GradientFeatureModule(c1)

        self.down2 = MicroDownsampleModule(c1, c2)
        self.mid   = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )

        self.down3 = MicroDownsampleModule(c2, c3)
        self.ctx   = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(c3, d, 0.1) for d in rates
        ])

    def forward(self, x):
        d1 = self.down1(x)
        d1 = d1 + self.grad(d1)     # gradient 강화
        d2 = self.mid(self.down2(d1))
        d3 = self.down3(d2)
        out = self.ctx(d3)
        return out, d2              # (low-res feature, skip)

# ───────────────────────────────────────────────────────────────
# 👇 4. 전체 네트워크 (Segmentation Head 포함)
# ───────────────────────────────────────────────────────────────
class submission_MiniNetV3_size(nn.Module):
    """
    MiniNetV3 (전체)
    Args
    ----
    in_ch       : 입력 채널(=3 RGB)
    num_classes : 출력 채널(클래스 수)
    ch          : (c1,c2,c3) 폭 설정 튜플
    interpolate : 마지막 출력 보간 사용 여부
    """
    def __init__(self,
                 in_ch: int,
                 num_classes: int,
                 ch: tuple = (6, 12, 18),
                 interpolate: bool = True):
        super().__init__()
        self.interpolate = interpolate
        c1, c2, c3 = ch

        # Encoder & Auxiliary Path
        self.encoder = MicroNetV5Encoder(in_ch, ch=ch)
        self.aux_ds  = MicroDownsampleModule(in_ch, c1)
        self.aux_ref = MicroResidualConvModule(c1, 1, 0.0)

        # Decoder
        self.up1     = MicroUpsampleModule(c3, c2)
        self.up_mid  = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )
        self.head    = nn.ConvTranspose2d(c2, num_classes,
                                          3, 2, 1, output_padding=1)

    def forward(self, x):
        aux  = self.aux_ref(self.aux_ds(x))
        enc, skip = self.encoder(x)

        y = self.up1(enc)
        if y.shape[2:] == skip.shape[2:]:
            y = y + skip
        if y.shape[2:] == aux.shape[2:]:
            y = y + aux

        y   = self.up_mid(y)
        out = self.head(y)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:],
                                mode="bilinear", align_corners=True)
        return out

# ───────────────────────────────────────────────────────────────
# 👇 5. 간단 테스트 (직접 실행 시)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 채널 폭 튜플 한 줄만 수정해서 실험!
    ch_cfg = (6, 12, 18)

    net = submission_MiniNetV3_size(in_ch=3, num_classes=21, ch=ch_cfg)
    x   = torch.randn(1, 3, 256, 256)
    y   = net(x)

    params = sum(p.numel() for p in net.parameters())
    print(f"출력 크기 : {y.shape}")
    print(f"파라미터 : {params/1e3:.2f} K")
