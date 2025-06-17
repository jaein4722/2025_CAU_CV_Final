"""
submission_MicroUNetv4s.py (Slim Version)
─────────────────────────
* v4 기반, 파라미터 감소 및 정확도 유지 목표
* 개선점
  ① base channel ↓ 14 → **12**          → 모델 전역 파라미터 감소
  ② MiniPPM 경량화 (출력 채널 c//4→c//8) → Context 블록 파라미터 대폭 감소
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Utility Blocks
# -------------------------------------------------
class DSConv(nn.Module):
    def __init__(self, c_in, c_out, use_residual=True, dilation=1):
        super().__init__()
        pad = dilation if isinstance(dilation, int) else dilation
        self.use_residual = use_residual and c_in == c_out
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=pad, dilation=dilation,
                            groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        y = self.bn(self.pw(self.dw(x)))
        if self.use_residual: y = y + x
        return self.act(y)

class InvertedResidual(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        mid = c*expand
        self.conv = nn.Sequential(
            nn.Conv2d(c, mid,1,bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid,mid,3,padding=1,groups=mid,bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid,c,1,bias=False), nn.BatchNorm2d(c))
    def forward(self,x): return x + self.conv(x)

class ECA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1,1,k, padding=k//2, bias=False)
    def forward(self,x):
        y = F.adaptive_avg_pool2d(x,1).squeeze(-1).transpose(1,2)
        y = torch.sigmoid(self.conv(y)).transpose(1,2).unsqueeze(-1)
        return x * y

# [V4s 개선점 ②]: MiniPPM 경량화
class MiniPPM_Slim(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s),
                          nn.Conv2d(c, c//8, 1, bias=False), # c//4 -> c//8
                          nn.ReLU(inplace=True))
            for s in (1,2,3,6)
        ])
        # 입력 채널 변경: c + 4*(c//8) = c + c/2
        self.fuse = nn.Conv2d(c + c//2, c, 1, bias=False)
    def forward(self,x):
        h,w = x.shape[2:]
        pools = [F.interpolate(s(x),(h,w),mode='nearest') for s in self.stages]
        return self.fuse(torch.cat([x]+pools, 1))

# -------------------------------------------------
# 2. Micro-UNet v4s
# -------------------------------------------------
class submission_MicroUNetv4s(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, use_residual=True):
        super().__init__()
        # [V4s 개선점 ①]: Base Channel 감소
        b = 12

        # Encoder
        self.enc1 = nn.Sequential(
            DSConv(in_channels,b, use_residual=False), 
            DSConv(b,b,use_residual=use_residual))
        self.enc2 = nn.Sequential(
            DSConv(b,b*2, use_residual=False), 
            DSConv(b*2,b*2,use_residual=use_residual))
        self.enc3 = nn.Sequential(
            DSConv(b*2,b*4, use_residual=False), 
            DSConv(b*4,b*4,use_residual=use_residual))
        self.pool = nn.MaxPool2d(2)
        self.eca_e2 = ECA(b*2); self.eca_e3 = ECA(b*4)

        # Context
        self.ctx = nn.Sequential(
            DSConv(b*4,b*4,dilation=2,use_residual=False),
            InvertedResidual(b*4),
            MiniPPM_Slim(b*4)) # 경량화된 MiniPPM_Slim 사용

        # Decoder
        self.up1_pre = DSConv(b*4,b*2, use_residual=False)
        self.fuse2   = nn.Conv2d(b*4,b*2,1,bias=False)
        self.eca_d2  = ECA(b*2)

        self.up2_pre = DSConv(b*2,b, use_residual=False)
        self.fuse1   = nn.Conv2d(b*2,b,1,bias=False)
        self.eca_d1  = ECA(b)

        # Head (br 제거)
        self.head = nn.Conv2d(b, num_classes,1)
        self._init()

    # -------------------------------------------------
    def forward(self,x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.eca_e2(self.enc2(p1))
        p2 = self.pool(e2)
        e3 = self.eca_e3(self.enc3(p2))

        # Context (Long residual connection)
        b_ctx = self.ctx(e3) + e3

        # Decoder
        d2 = F.interpolate(b_ctx, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up1_pre(d2)
        d2 = self.fuse2(torch.cat([d2,e2],1))
        d2 = self.eca_d2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up2_pre(d1)
        d1 = self.fuse1(torch.cat([d1,e1],1))
        d1 = self.eca_d1(d1)

        # Head
        logits = self.head(d1)
        return logits

    # -------------------------------------------------
    def _init(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1.); nn.init.constant_(m.bias,0.)

# -------------------------------------------------
# 3. Quick Test
# -------------------------------------------------
if __name__ == "__main__":
    net = submission_MicroUNetv4s(in_channels=3, num_classes=21)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters())
    print(f"Params: {p/1e3:.1f} K")