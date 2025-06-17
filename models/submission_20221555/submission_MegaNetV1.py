import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# MegaNetV1: 0.53+ IoU 목표 달성을 위한 최고 성능 세그멘테이션 모델
# 모든 혁신 기술의 시너지를 활용한 하이브리드 아키텍처
# =============================================================================

class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution - 기본 효율 모듈"""
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
    """Multi-dilation separable conv - MiniNetV3의 핵심 기법"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        padding2 = padding + (dilation - 1) * (kernel_size - 1) // 2
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, groups=in_channels, bias=False)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding2, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
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

class CoordinateAttention(nn.Module):
    """위치 정보 보존 어텐션 - 성능 향상의 핵심"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip, eps=1e-3)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        return out

class MultiGradientModule(nn.Module):
    """다중 그래디언트 검출 모듈 - Edge 강화"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Sobel X, Y 필터
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Laplacian 필터
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)
        
        # DoG (Difference of Gaussians) 시뮬레이션
        dog = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).expand(in_channels, 1, 3, 3)
        self.register_buffer('dog', dog)
        
        # Feature refinement
        self.refine_conv = SeparableConv2d(in_channels * 4, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        laplace = F.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])
        dog_out = F.conv2d(x, self.dog, padding=1, groups=x.shape[1])
        
        gradients = torch.cat([grad_x, grad_y, laplace, dog_out], dim=1)
        out = self.refine_conv(gradients)
        out = self.bn(out)
        return F.relu(out)

class AdvancedASPP(nn.Module):
    """향상된 Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # ASPP branches with extended dilation rates
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//6, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//6, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = SeparableConv2d(in_channels, out_channels//6, kernel_size=3, padding=6, dilation=6, bias=False)
        self.branch3 = SeparableConv2d(in_channels, out_channels//6, kernel_size=3, padding=12, dilation=12, bias=False)
        self.branch4 = SeparableConv2d(in_channels, out_channels//6, kernel_size=3, padding=18, dilation=18, bias=False)
        self.branch5 = SeparableConv2d(in_channels, out_channels//6, kernel_size=3, padding=24, dilation=24, bias=False)
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//6, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//6, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Coordinate attention for fusion
        self.attention = CoordinateAttention(out_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)
        
        # Global pooling and upsample
        bg = self.global_pool(x)
        bg = F.interpolate(bg, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        concat = torch.cat([b1, b2, b3, b4, b5, bg], dim=1)
        fused = self.fusion(concat)
        
        # Apply coordinate attention
        out = self.attention(fused)
        
        return out

class DatasetAdaptiveModule(nn.Module):
    """데이터셋별 특화 처리 모듈"""
    def __init__(self, channels):
        super().__init__()
        
        # VOC용: 큰 receptive field, 복잡한 객체
        self.voc_branch = nn.Sequential(
            SeparableConv2d(channels, channels//2, kernel_size=3, padding=6, dilation=6),
            SeparableConv2d(channels//2, channels//2, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(channels//2, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Medical용: 세밀한 boundary detection
        self.medical_branch = nn.Sequential(
            SeparableConv2d(channels, channels//2, kernel_size=3, padding=1, dilation=1),
            SeparableConv2d(channels//2, channels//2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels//2, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # 융합 및 선택
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.attention = CoordinateAttention(channels)

    def forward(self, x):
        voc_feat = self.voc_branch(x)
        med_feat = self.medical_branch(x)
        
        # Adaptive combination
        combined = torch.cat([voc_feat, med_feat], dim=1)
        fused = self.fusion(combined)
        
        # Apply attention
        out = self.attention(fused)
        
        return out

class MegaDownsampleModule(nn.Module):
    """향상된 다운샘플링 모듈"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.use_maxpool = in_channels < out_channels
        
        if not self.use_maxpool:
            channels_conv = out_channels
        else:
            channels_conv = out_channels - in_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels_conv, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels_conv, eps=1e-3)
        )
        
        self.bn_final = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        
        if self.use_maxpool:
            x_pool = F.max_pool2d(x, kernel_size=2, stride=2)
            out = torch.cat([out, x_pool], dim=1)
        
        out = self.bn_final(out)
        return self.act(out)

class MegaResidualBlock(nn.Module):
    """강화된 Residual 블록"""
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = MultiDilationSeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)
        
        self.conv2 = SeparableConv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)
        
        self.dropout = nn.Dropout2d(dropout)
        self.attention = CoordinateAttention(channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = out + identity
        out = self.attention(out)
        out = F.relu(out)
        
        return out

class MegaEncoder(nn.Module):
    """MegaNetV1 인코더 - 최고 성능 추구"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Progressive channel expansion: 3 → 20 → 40 → 64 → 96
        self.downsample_1 = MegaDownsampleModule(in_channels, 20)
        self.gradient_1 = MultiGradientModule(20)
        
        self.downsample_2 = MegaDownsampleModule(20, 40)
        self.blocks_2 = nn.Sequential(*[
            MegaResidualBlock(40, dilation=1, dropout=0.05),
            MegaResidualBlock(40, dilation=2, dropout=0.05)
        ])
        
        self.downsample_3 = MegaDownsampleModule(40, 64)
        self.blocks_3 = nn.Sequential(*[
            MegaResidualBlock(64, dilation=1, dropout=0.1),
            MegaResidualBlock(64, dilation=2, dropout=0.1),
            MegaResidualBlock(64, dilation=4, dropout=0.1)
        ])
        
        self.downsample_4 = MegaDownsampleModule(64, 96)
        
        # Multi-scale feature extraction
        rates = [1, 2, 4, 6, 8, 12]
        self.feature_modules = nn.Sequential(*[
            MegaResidualBlock(96, dilation=rate, dropout=0.15) for rate in rates
        ])
        
        # Advanced ASPP
        self.aspp = AdvancedASPP(96, 96)
        
        # Dataset adaptive processing
        self.dataset_adaptive = DatasetAdaptiveModule(96)

    def forward(self, x):
        # Stage 1
        d1 = self.downsample_1(x)
        d1_grad = self.gradient_1(d1)
        d1_enhanced = d1 + d1_grad
        
        # Stage 2
        d2 = self.downsample_2(d1_enhanced)
        d2 = self.blocks_2(d2)
        
        # Stage 3
        d3 = self.downsample_3(d2)
        d3 = self.blocks_3(d3)
        
        # Stage 4
        d4 = self.downsample_4(d3)
        d4 = self.feature_modules(d4)
        
        # Advanced processing
        d4_aspp = self.aspp(d4)
        d4_adaptive = self.dataset_adaptive(d4_aspp)
        
        return d4_adaptive, d3, d2, d1_enhanced

class AdvancedFPNDecoder(nn.Module):
    """FPN + PANet 하이브리드 디코더"""
    def __init__(self, feature_channels=[96, 64, 40, 20], out_channels=32):
        super().__init__()
        
        # Top-down pathway (FPN)
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(feature_channels[0], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(feature_channels[1], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(feature_channels[2], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(feature_channels[3], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Refinement modules
        self.refine1 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        self.refine2 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        self.refine3 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        
        # Bottom-up pathway (PANet)
        self.panet_conv1 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        self.panet_conv2 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        self.panet_conv3 = MegaResidualBlock(out_channels, dilation=1, dropout=0.1)
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
            CoordinateAttention(out_channels)
        )

    def forward(self, features):
        f4, f3, f2, f1 = features
        
        # Top-down pathway (FPN)
        p4 = self.fpn_conv1(f4)
        
        p3 = self.fpn_conv2(f3)
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.refine1(p3)
        
        p2 = self.fpn_conv3(f2)
        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.refine2(p2)
        
        p1 = self.fpn_conv4(f1)
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.refine3(p1)
        
        # Bottom-up pathway (PANet)
        n1 = self.panet_conv1(p1)
        
        n2 = F.max_pool2d(n1, kernel_size=2, stride=2)
        n2 = n2 + p2
        n2 = self.panet_conv2(n2)
        
        n3 = F.max_pool2d(n2, kernel_size=2, stride=2)
        n3 = n3 + p3
        n3 = self.panet_conv3(n3)
        
        # Multi-scale fusion
        n1_up = n1  # Already at target size
        n2_up = F.interpolate(n2, size=n1.shape[2:], mode='bilinear', align_corners=True)
        n3_up = F.interpolate(n3, size=n1.shape[2:], mode='bilinear', align_corners=True)
        p4_up = F.interpolate(p4, size=n1.shape[2:], mode='bilinear', align_corners=True)
        
        # Final fusion
        fused = torch.cat([n1_up, n2_up, n3_up, p4_up], dim=1)
        out = self.final_fusion(fused)
        
        return out

class submission_MegaNetV1(nn.Module):
    """
    MegaNetV1: 0.53+ IoU 목표 달성을 위한 최고 성능 세그멘테이션 모델
    
    핵심 기술:
    - CoordinateAttention: 위치 정보 보존
    - AdvancedASPP: 확장된 다중 스케일 처리
    - MultiGradientModule: 다양한 edge 검출
    - DatasetAdaptiveModule: 데이터셋별 특화 처리
    - AdvancedFPNDecoder: FPN + PANet 하이브리드
    """
    def __init__(self, in_channels, num_classes, interpolate=True):
        super().__init__()
        
        self.interpolate = interpolate
        
        # Mega encoder with all advanced features
        self.encoder = MegaEncoder(in_channels)
        
        # Enhanced auxiliary path for low-level features
        self.aux_downsample = MegaDownsampleModule(in_channels, 20)
        self.aux_gradient = MultiGradientModule(20)
        self.aux_refine = nn.Sequential(
            MegaResidualBlock(20, dilation=1, dropout=0.05),
            CoordinateAttention(20)
        )
        
        # Advanced FPN decoder
        self.decoder = AdvancedFPNDecoder(
            feature_channels=[96, 64, 40, 20],
            out_channels=48
        )
        
        # Final output modules
        self.final_refine = nn.Sequential(
            MegaResidualBlock(48, dilation=1, dropout=0.1),
            MegaResidualBlock(48, dilation=2, dropout=0.1),
            CoordinateAttention(48)
        )
        
        # Output head with multi-scale supervision
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1, bias=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Enhanced auxiliary path
        aux = self.aux_downsample(x)
        aux_grad = self.aux_gradient(aux)
        aux_enhanced = aux + aux_grad
        aux_refined = self.aux_refine(aux_enhanced)
        
        # Main encoder path
        features = self.encoder(x)
        f4, f3, f2, f1 = features
        
        # Integrate auxiliary features
        if aux_refined.shape[2:] == f1.shape[2:]:
            f1 = f1 + aux_refined
        
        # Advanced FPN decoding
        decoded = self.decoder([f4, f3, f2, f1])
        
        # Final refinement
        refined = self.final_refine(decoded)
        
        # Output
        out = self.output_conv(refined)
        
        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out

if __name__ == "__main__":
    # Test model
    print("=== MegaNetV1 Model Test ===")
    
    # Test with different configurations
    model_voc = submission_MegaNetV1(in_channels=3, num_classes=21)
    model_binary = submission_MegaNetV1(in_channels=3, num_classes=2)
    
    # Parameter count
    params_voc = sum(p.numel() for p in model_voc.parameters() if p.requires_grad)
    params_binary = sum(p.numel() for p in model_binary.parameters() if p.requires_grad)
    
    print(f"VOC (21 classes): {params_voc:,} parameters")
    print(f"Binary (2 classes): {params_binary:,} parameters")
    print(f"Average: {(params_voc + params_binary*4)/5:,.0f} parameters")
    
    # Forward pass test
    x_test = torch.randn(1, 3, 256, 256)
    
    # Set to eval mode to avoid BatchNorm issues with batch_size=1
    model_voc.eval()
    model_binary.eval()
    
    with torch.no_grad():
        out_voc = model_voc(x_test)
        out_binary = model_binary(x_test)
    
    print(f"VOC output shape: {out_voc.shape}")
    print(f"Binary output shape: {out_binary.shape}")
    print("✅ MegaNetV1 ready for training!") 