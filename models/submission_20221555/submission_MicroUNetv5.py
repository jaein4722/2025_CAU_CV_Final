"""
submission_MicroUNetv5.py
─────────────────────────
* CFD 성능 개선 목표 버전
* 개선점 (v4 대비)
  ① Stem 레이어 도입: MaxPool을 제거하고 원본 해상도에서 특징을 먼저 추출하여 
     얇은 균열(crack) 정보의 초기 손실을 최소화.
  ② 학습 가능한 다운샘플링: MaxPool2d를 stride=2 Conv2d로 대체하여 
     중요 특징을 보존하는 방향으로 다운샘플링을 학습.
  ③ 컨텍스트 블록 강화: 다양한 Dilation을 가진 DSConv를 추가하여 
     여러 스케일의 길쭉한 특징 포착 능력을 향상 (LiteASPP 아이디어 차용).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Utility Blocks (v4와 동일)
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

class MiniPPM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s),
                          nn.Conv2d(c, c//4, 1, bias=False),
                          nn.ReLU(inplace=True))
            for s in (1,2,3,6)
        ])
        self.fuse = nn.Conv2d(c + c, c, 1, bias=False)
    def forward(self,x):
        h,w = x.shape[2:]
        pools = [F.interpolate(s(x),(h,w),mode='nearest') for s in self.stages]
        return self.fuse(torch.cat([x]+pools, 1))

# -------------------------------------------------
# 2. Micro-UNet v5 (CFD 성능 개선 목표)
# -------------------------------------------------
class submission_MicroUNetv5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, use_residual=True):
        super().__init__()
        b = 14  # base channel

        # -------- Stem & Encoder --------
        # [V5 변경점 ①]: 초기 MaxPool 제거, 원본 해상도에서 특징 추출하는 Stem 레이어
        self.stem = nn.Sequential(
            DSConv(in_channels, b, use_residual=False), 
            DSConv(b, b, use_residual=use_residual)
        )
        
        # [V5 변경점 ②]: MaxPool을 학습 가능한 Strided Conv로 대체
        self.down1 = nn.Conv2d(b, b*2, kernel_size=2, stride=2, bias=False)
        self.enc2 = nn.Sequential(
            DSConv(b*2, b*2, use_residual=False), # 채널 수가 바뀌므로 residual=False
            DSConv(b*2, b*2, use_residual=use_residual)
        )
        
        self.down2 = nn.Conv2d(b*2, b*4, kernel_size=2, stride=2, bias=False)
        self.enc3 = nn.Sequential(
            DSConv(b*4, b*4, use_residual=False), # 채널 수가 바뀌므로 residual=False
            DSConv(b*4, b*4, use_residual=use_residual)
        )
        self.eca_e2 = ECA(b*2); self.eca_e3 = ECA(b*4)

        # -------- Context --------
        # [V5 변경점 ③]: Dilated Conv 추가하여 Context 블록 강화
        self.ctx = nn.Sequential(
            DSConv(b*4, b*4, dilation=1, use_residual=False),
            DSConv(b*4, b*4, dilation=2, use_residual=False),
            DSConv(b*4, b*4, dilation=4, use_residual=False),
            InvertedResidual(b*4),
            MiniPPM(b*4)
        )

        # -------- Decoder (v4와 구조 동일) --------
        self.up1_pre = DSConv(b*4, b*2, use_residual=False)
        self.fuse2   = nn.Conv2d(b*4, b*2, 1, bias=False)
        self.eca_d2  = ECA(b*2)

        self.up2_pre = DSConv(b*2, b, use_residual=False)
        self.fuse1   = nn.Conv2d(b*2, b, 1, bias=False)
        self.eca_d1  = ECA(b)

        # -------- Head (v4와 구조 동일) --------
        self.head = nn.Conv2d(b, num_classes, 1)
        self.br   = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        self._init()

    # -------------------------------------------------
    def forward(self,x):
        # Stem & Encoder
        s1 = self.stem(x)                   # (B, 14, H, W)
        
        s2_in = self.down1(s1)              # (B, 28, H/2, W/2)
        s2 = self.eca_e2(self.enc2(s2_in))
        
        s3_in = self.down2(s2)              # (B, 56, H/4, W/4)
        s3 = self.eca_e3(self.enc3(s3_in))

        # Context
        b  = self.ctx(s3) + s3              # Long residual connection

        # Decoder
        d2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up1_pre(d2)
        d2 = self.fuse2(torch.cat([d2, s2], 1)) # Skip connection from s2
        d2 = self.eca_d2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up2_pre(d1)
        d1 = self.fuse1(torch.cat([d1, s1], 1)) # Skip connection from s1
        d1 = self.eca_d1(d1)

        # Head
        logits = self.head(d1)
        #logits = logits + self.br(logits)
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
    net = submission_MicroUNetv5(in_channels=3, num_classes=21)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("Output shape:", y.shape)
    p = sum(p.numel() for p in net.parameters())
    print(f"Params: {p/1e3:.1f} K") # 파라미터 수는 약간 증가합니다.