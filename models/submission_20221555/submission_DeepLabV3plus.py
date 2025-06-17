import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# 1. Building Blocks (DSConv, InvertedResidual)
# -------------------------------------------------
class DSConv(nn.Module):
    """기본 빌딩 블록: Depthwise Separable Convolution"""
    def __init__(self, c_in, c_out, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=dilation, 
                                   dilation=dilation, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class InvertedResidual(nn.Module):
    """MobileNetV2의 핵심 블록. 채널을 확장했다가 다시 줄여 표현력 증대"""
    def __init__(self, c_in, c_out, stride, expand_ratio):
        super().__init__()
        self.use_residual = stride == 1 and c_in == c_out
        hidden_dim = int(round(c_in * expand_ratio))
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(c_in, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])

        # Pointwise linear
        layers.append(nn.Conv2d(hidden_dim, c_out, 1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -------------------------------------------------
# 2. DeepLabV3+ Components (ASPP, Decoder)
# -------------------------------------------------
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: 다양한 스케일의 Context 포착"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 6, 12, 18]
        
        # 1x1 Conv, 3x3 Dilated Convs
        self.aspp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ) if d == 1 else nn.Sequential(
                DSConv(in_channels, out_channels, dilation=d), # 3x3 conv을 DSConv로 경량화
            ) for d in dilations
        ])
        
        # Global Average Pooling Branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            
        # 모든 특징을 합친 후 최종 Conv
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        size = x.shape[2:]
        branches = [conv(x) for conv in self.aspp_convs]
        
        pool_branch = self.global_pool(x)
        pool_branch = F.interpolate(pool_branch, size, mode='bilinear', align_corners=False)
        branches.append(pool_branch)
        
        concatenated = torch.cat(branches, dim=1)
        return self.fuse(concatenated)

# -------------------------------------------------
# 3. The Final Model: submission_VOC_Specialist
# -------------------------------------------------
class submission_DeepLabV3plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        # VOC 정복을 위해 체급을 올린 설정 (약 1.7M 파라미터)
        # InvertedResidual 설정: t(확장비), c(출력채널), n(반복횟수), s(stride)
        backbone_settings = [
            [1, 16, 1, 1],  # stage 0
            [6, 24, 2, 2],  # stage 1 (Low-level features for decoder)
            [6, 40, 2, 2],  # stage 2
            [6, 80, 3, 2],  # stage 3
            [6, 160, 3, 1]  # stage 4
        ]
        
        # --- Backbone (Encoder) ---
        input_channel = 32
        self.stem = nn.Sequential(DSConv(in_channels, input_channel, stride=2), DSConv(input_channel, input_channel))
        
        self.stages = nn.ModuleList()
        for t, c, n, s in backbone_settings:
            stage = []
            for i in range(n):
                stride = s if i == 0 else 1
                stage.append(InvertedResidual(input_channel, c, stride, expand_ratio=t))
                input_channel = c
            self.stages.append(nn.Sequential(*stage))
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # --- ASPP ---
        self.aspp = ASPP(in_channels=256, out_channels=256)

        # --- Decoder ---
        # 저수준 특징(low_level_features) 처리
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False), # stage 1의 출력 채널은 24
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
            
        self.decoder_fuse = nn.Sequential(
            DSConv(304, 256), # 256(from ASPP) + 48(from low_level) = 304
            nn.Dropout(0.5),
            DSConv(256, 256))
            
        self.final_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.stages[0](x)
        low_level_features = self.stages[1](x) # 저수준 특징 저장 (1/4 크기)
        x = self.stages[2](low_level_features)
        x = self.stages[3](x)
        x = self.stages[4](x)
        x = self.last_conv(x) # 최종 백본 특징 (1/16 크기)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        # ASPP 특징을 4배 업샘플링하여 저수준 특징과 크기를 맞춤
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        low_level_features = self.low_level_conv(low_level_features)
        
        # 융합
        x = torch.cat([x, low_level_features], dim=1)
        x = self.decoder_fuse(x)
        
        # 최종 출력
        x = self.final_conv(x)
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

if __name__ == "__main__":
    net = submission_DeepLabV3plus(in_channels=3, num_classes=21)
    p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model: VOC_Specialist, Params: {p/1e6:.2f}M")
    # 예상 파라미터: 약 1.70M. 300만 제한 내에서 매우 강력한 체급.