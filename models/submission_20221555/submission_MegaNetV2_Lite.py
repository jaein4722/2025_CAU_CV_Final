import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MegaNetV2_Lite: MegaNetV1의 핵심 기술만 추출한 경량화 모델
# 목표: 50k 파라미터 이하로 0.47+ IoU 달성
# =============================================================================

class MultiDilationSeparableConv2d(nn.Module):
    """MiniNetV3의 핵심 기법 - 검증된 효과"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)
        out = x1 + x2  # 두 dilation의 합
        out = self.pointwise(out)
        out = self.bn(out)
        return F.relu(out)

class LightweightGradientModule(nn.Module):
    """경량화된 Edge Enhancement - Sobel만 사용"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Sobel X, Y 필터만 사용 (가장 효과적)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 간단한 융합
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        
        gradients = torch.cat([grad_x, grad_y], dim=1)
        out = self.fusion(gradients)
        out = self.bn(out)
        return F.relu(out)

class CompactCoordinateAttention(nn.Module):
    """경량화된 Coordinate Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(4, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = F.relu(self.conv1(y))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_w * a_h

class CompactASPP(nn.Module):
    """경량화된 ASPP - 3개 branch만 사용"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 3개 branch로 축소
        self.branch1 = nn.Conv2d(in_channels, out_channels//3, kernel_size=1, bias=False)
        
        # Depthwise separable convolution for dilation branches
        self.branch2_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=6, dilation=6, groups=in_channels, bias=False)
        self.branch2_pw = nn.Conv2d(in_channels, out_channels//3, kernel_size=1, bias=False)
        
        self.branch3_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=12, dilation=12, groups=in_channels, bias=False)
        self.branch3_pw = nn.Conv2d(in_channels, out_channels//3, kernel_size=1, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2_pw(self.branch2_dw(x)))
        b3 = F.relu(self.branch3_pw(self.branch3_dw(x)))
        
        concat = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(concat)

class VOCSpecializedHead(nn.Module):
    """VOC 데이터셋 성능 개선을 위한 전용 헤드"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # VOC는 21개 클래스로 복잡하므로 더 강력한 분류기 필요
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels//2, num_classes, kernel_size=1)
        )
        
        # Auxiliary classifier for better gradient flow
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        main_out = self.classifier(x)
        aux_out = self.aux_classifier(x)
        return main_out, aux_out

class LiteEncoder(nn.Module):
    """경량화된 인코더 - 채널 수 대폭 축소"""
    def __init__(self, in_channels):
        super().__init__()
        
        # 채널 수를 대폭 축소: 3→16→24→32→40
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 16→24
        self.stage1 = nn.Sequential(
            MultiDilationSeparableConv2d(16, 24, dilation=2),
            LightweightGradientModule(24),
            CompactCoordinateAttention(24)
        )
        
        # Stage 2: 24→32  
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True),
            MultiDilationSeparableConv2d(32, 32, dilation=3),
            CompactCoordinateAttention(32)
        )
        
        # Stage 3: 32→40
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40, eps=1e-3),
            nn.ReLU(inplace=True),
            CompactASPP(40, 40),
            CompactCoordinateAttention(40)
        )

    def forward(self, x):
        x = self.stem(x)  # /2
        
        f1 = self.stage1(x)  # /2, 24 channels
        f2 = self.stage2(f1)  # /4, 32 channels  
        f3 = self.stage3(f2)  # /8, 40 channels
        
        return [f1, f2, f3]

class LiteDecoder(nn.Module):
    """경량화된 디코더"""
    def __init__(self, feature_channels=[24, 32, 40], out_channels=32):
        super().__init__()
        
        # 간단한 FPN 구조
        self.lateral3 = nn.Conv2d(feature_channels[2], out_channels, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(feature_channels[1], out_channels, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1, bias=False)
        
        # 융합 레이어들
        self.smooth3 = MultiDilationSeparableConv2d(out_channels, out_channels, dilation=2)
        self.smooth2 = MultiDilationSeparableConv2d(out_channels, out_channels, dilation=2)
        self.smooth1 = MultiDilationSeparableConv2d(out_channels, out_channels, dilation=1)
        
        # 최종 융합
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        f1, f2, f3 = features
        
        # Lateral connections
        p3 = self.lateral3(f3)
        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.lateral1(f1) + F.interpolate(p2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        
        # Smooth
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        
        # 모든 레벨을 같은 크기로 업샘플링하여 융합
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=True)
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=True)
        
        # 융합
        fused = torch.cat([p1, p2_up, p3_up], dim=1)
        out = self.final_fusion(fused)
        
        return out

class submission_MegaNetV2_Lite(nn.Module):
    """MegaNetV1의 핵심 기술만 추출한 경량화 모델"""
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.encoder = LiteEncoder(in_channels)
        self.decoder = LiteDecoder()
        
        # 일반 분류 헤드 (2클래스용)
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
        # VOC 전용 헤드 (21클래스용)
        self.voc_head = VOCSpecializedHead(32, 21)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder  
        decoded = self.decoder(features)
        
        # 클래스 수에 따라 다른 헤드 사용
        if hasattr(self, 'training') and self.training:
            # 훈련 시에는 num_classes를 알 수 없으므로 일반 헤드 사용
            out = self.classifier(decoded)
        else:
            # 추론 시 클래스 수 확인
            if decoded.shape[0] > 0:  # 배치가 있는 경우
                # VOC 데이터셋 감지 (21클래스)
                out = self.classifier(decoded)
                if out.shape[1] == 21:  # VOC 데이터셋
                    main_out, aux_out = self.voc_head(decoded)
                    out = main_out + 0.4 * aux_out  # Auxiliary loss 가중치
            else:
                out = self.classifier(decoded)
        
        # 원본 크기로 업샘플링
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out 