import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 핵심 빌딩 블록 (BN-ReLU 순서, MultiDilation 기본값 유지) ---

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x = self.dw(x)
        x = F.relu(self.bn(x))
        x = self.pw(x)
        return x

class MultiDilationSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=2, bias=True): # s=1, p=1, d=2 기본값 유지
        super().__init__()
        p1 = p
        p2 = p + (d - 1) * (k - 1) // 2

        self.dw1 = nn.Conv2d(in_ch, in_ch, k, s, p1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3)

        self.dw2 = nn.Conv2d(in_ch, in_ch, k, s, p2, d, groups=in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch, eps=1e-3)

        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.dw1(x)))
        x2 = F.relu(self.bn2(self.dw2(x)))
        out = x1 + x2
        return self.pw(out)

class MicroDownsampleModule(nn.Module):
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
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        # k=3, s=1, p=dil, d=dil 유지
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, False) 
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(self.bn(y))
        y = self.drop(y)
        return F.relu(x + y)

class MicroResidualMultiDilationConvModule(nn.Module):
    def __init__(self, ch, dil=2, drop=0.): # dil 기본값 2 유지
        super().__init__()
        # k=3, s=1, p=1, d=dil 유지
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, 1, 1, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(self.bn(y))
        y = self.drop(y)
        return F.relu(x + y)

class MicroUpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, output_padding=1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)))

# --- 2. GradientFeatureModule (채널 제어 유지) ---

class GradientFeatureModule(nn.Module):
    # GradientFeatureModule은 in_ch와 out_ch_refine을 명시적으로 받도록 수정.
    # 이렇게 하면 GradientFeatureModule의 내부 채널을 유연하게 조절 가능.
    def __init__(self, in_ch, out_ch_refine):
        super().__init__()
        # Sobel 필터는 입력 채널 수만큼 복제되어 각 채널에 독립적으로 적용
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_ch, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=torch.float32).expand(in_ch, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # SeparableConv2d의 in_ch는 gx, gy를 concatenate한 ch * 2, out_ch는 out_ch_refine
        self.refine = SeparableConv2d(in_ch * 2, out_ch_refine, 1, 1, 0, 1, False)
        self.bn     = nn.BatchNorm2d(out_ch_refine, eps=1e-3)

    def forward(self, x):
        # groups=x.size(1)을 통해 각 입력 채널에 대해 독립적으로 소벨 필터 적용
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        g  = torch.cat([gx, gy], dim=1) # (Batch, in_ch * 2, H, W)
        return F.relu(self.bn(self.refine(g))) # (Batch, out_ch_refine, H, W)

# --- 3. 인코더 구조 (채널 극소화 및 GradientFeatureModule 통합 재조정) ---

class MicroNetV5Encoder(nn.Module):
    # 채널을 (6, 12, 18)로 극소화
    def __init__(self, in_ch: int, ch: tuple = (6, 12, 18), rates: tuple = (1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3 = ch # (6, 12, 18)

        # GradientFeatureModule은 in_ch를 그대로 받아서, out_ch를 c1과 동일하게 만듦
        # 이렇게 하면 d1의 결과와 합쳐질 때 채널이 일치함
        self.grad  = GradientFeatureModule(in_ch, c1)

        # Downsample1은 in_ch를 받아서 c1으로 줄임
        self.down1 = MicroDownsampleModule(in_ch, c1) 

        # Downsample2는 c1을 받아서 c2로 줄임
        self.down2 = MicroDownsampleModule(c1, c2)
        self.mid   = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )

        # Downsample3는 c2를 받아서 c3로 줄임
        self.down3 = MicroDownsampleModule(c2, c3)
        self.ctx   = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(c3, d, 0.1) for d in rates
        ])

    def forward(self, x):
        # 1. 초기 다운샘플링 (in_ch -> c1, H -> H/2)
        d1 = self.down1(x) # (Batch, c1, H/2, W/2)

        # 2. GradientFeatureModule 적용 (원본 x에서 그라디언트 특징 추출)
        # GradientFeatureModule은 (Batch, in_ch, H, W) -> (Batch, c1, H, W)
        g_feat = self.grad(x) 

        # 3. g_feat의 해상도를 d1과 맞춤 (H/2, W/2)
        if g_feat.shape[2:] != d1.shape[2:]:
            g_feat = F.avg_pool2d(g_feat, kernel_size=2, stride=2) # H/2, W/2로 강제 리사이즈

        # 4. d1에 형태 특징을 더함
        d1_enhanced = d1 + g_feat # (Batch, c1, H/2, W/2)

        # 5. 다음 다운샘플링 및 중간 모듈 (c1 -> c2, H/2 -> H/4)
        d2 = self.mid(self.down2(d1_enhanced)) # (Batch, c2, H/4, W/4) -> skip

        # 6. 마지막 다운샘플링 및 컨텍스트 모듈 (c2 -> c3, H/4 -> H/8)
        d3 = self.down3(d2) # (Batch, c3, H/8, W/8)
        out = self.ctx(d3) # (Batch, c3, H/8, W/8) -> enc

        return out, d2 # enc, skip

# --- 4. 최종 MicroNetv5 모델 (submission_20221377) ---

class submission_final(nn.Module):
    # 채널을 (6, 12, 18)로 극소화
    def __init__(self, in_ch: int, num_classes: int, ch: tuple = (6, 12, 18), interpolate: bool = True):
        super().__init__()
        self.interpolate = interpolate
        c1, c2, c3 = ch # (6, 12, 18)

        self.encoder = MicroNetV5Encoder(in_ch, ch=ch)

        # Auxiliary Path: in_ch -> c2 (12채널)로 변경
        # 인코더의 skip (d2)와 동일한 채널(c2) 및 해상도(H/4)를 갖도록 조정
        self.aux_ds  = MicroDownsampleModule(in_ch, c2) # in_ch -> c2 (12채널)
        self.aux_ref = MicroResidualConvModule(c2, 1, 0.0) # c2 (12채널)

        # 디코더 업샘플링: c3 -> c2 (18 -> 12)
        self.up1     = MicroUpsampleModule(c3, c2) 
        self.up_mid  = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0), # c2 (12채널)
            MicroResidualConvModule(c2, 1, 0.0)
        )
        # 최종 출력: c2 -> num_classes (12 -> num_classes)
        self.head    = nn.ConvTranspose2d(c2, num_classes, 3, 2, 1, output_padding=1)

    def forward(self, x):
        # Auxiliary Path: (Batch, in_ch, H, W) -> (Batch, c2, H/2, W/2) -> (Batch, c2, H/4, W/4)
        aux_raw = self.aux_ds(x) # (Batch, c2, H/2, W/2)
        aux     = self.aux_ref(aux_raw) # (Batch, c2, H/2, W/2)

        enc, skip = self.encoder(x) # enc: (Batch, c3, H/8, W/8), skip: (Batch, c2, H/4, W/4)

        # 첫 번째 업샘플링: enc (H/8) -> y (H/4)
        y = self.up1(enc) # (Batch, c2, H/4, W/4)

        # skip (d2)와 aux (H/2)를 y (H/4)에 더하기 전에 해상도와 채널 확인
        # skip은 (H/4,W/4,c2)이므로 바로 더할 수 있음
        if y.shape[2:] == skip.shape[2:] and y.shape[1] == skip.shape[1]:
            y = y + skip

        # aux는 (H/2,W/2,c2)이므로 H/4,W/4로 avg_pool 해야 함
        aux_pooled = F.avg_pool2d(aux, kernel_size=2, stride=2)
        if y.shape[2:] == aux_pooled.shape[2:] and y.shape[1] == aux_pooled.shape[1]:
            y = y + aux_pooled

        y   = self.up_mid(y) # (Batch, c2, H/4, W/4)
        out = self.head(y) # (Batch, num_classes, H/2, W/2)

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return out