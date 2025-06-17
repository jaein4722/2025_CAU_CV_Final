import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Cartesian Decomposition Modules (LeMoRe 기반) ---

class CartesianEncoder(nn.Module):
    """3개 직교 방향으로 feature 분해하는 혁신적 인코더"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Efficient parameter sharing
        reduced_ch = max(4, out_channels // 4)
        
        # Cartesian Views: Transverse, Frontal, Lateral
        self.transverse_conv = nn.Conv2d(in_channels, reduced_ch, 1, bias=False)
        self.frontal_conv = nn.Conv2d(in_channels, reduced_ch, 1, bias=False) 
        self.lateral_conv = nn.Conv2d(in_channels, reduced_ch, 1, bias=False)
        
        # Fusion and refinement
        self.fusion_conv = nn.Conv2d(reduced_ch * 3, out_channels, 1, bias=False)
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        # Extract orthogonal views
        trans_view = self.transverse_conv(x)  # Transverse projection
        front_view = self.frontal_conv(x.transpose(-2, -1)).transpose(-2, -1)  # Frontal
        lat_view = self.lateral_conv(x.flip(-1)).flip(-1)  # Lateral
        
        # Fuse multi-dimensional views
        fused = torch.cat([trans_view, front_view, lat_view], dim=1)
        out = self.fusion_conv(fused)
        out = self.dwconv(out)
        out = self.act(self.bn(out))
        
        return out

class NestedAttention(nn.Module):
    """경량화된 Nested Attention (LeMoRe 방식)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_ch = max(1, channels // reduction)
        
        # 3개 Q, K, V pairs
        self.q_convs = nn.ModuleList([
            nn.Conv2d(channels, reduced_ch, 1, bias=False) for _ in range(3)
        ])
        self.k_convs = nn.ModuleList([
            nn.Conv2d(channels, reduced_ch, 1, bias=False) for _ in range(3)
        ])
        self.v_convs = nn.ModuleList([
            nn.Conv2d(channels, reduced_ch, 1, bias=False) for _ in range(3)
        ])
        
        self.output_conv = nn.Conv2d(reduced_ch * 3, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V triplets
        qs = [conv(x) for conv in self.q_convs]
        ks = [conv(x) for conv in self.k_convs] 
        vs = [conv(x) for conv in self.v_convs]
        
        # Nested attention computation
        attended_features = []
        for i in range(3):
            q = qs[i].view(B, -1, H*W)  # B, C//4, HW
            
            # Cross-attention with all keys
            attn_sum = 0
            for j in range(3):
                k = ks[j].view(B, -1, H*W)  # B, C//4, HW
                attn = torch.bmm(q.transpose(1, 2), k) / (q.size(1) ** 0.5)  # B, HW, HW
                attn = F.softmax(attn, dim=-1)
                
                v = vs[j].view(B, -1, H*W)  # B, C//4, HW
                attn_sum += torch.bmm(v, attn.transpose(1, 2))  # B, C//4, HW
            
            attended_features.append(attn_sum.view(B, -1, H, W))
        
        # Combine all attended features
        combined = torch.cat(attended_features, dim=1)
        out = self.output_conv(combined)
        
        return x * self.sigmoid(out)

# --- Dataset-Specific Modules ---

class DatasetAwareModule(nn.Module):
    """데이터셋별 특화 처리 모듈"""
    def __init__(self, channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        if num_classes == 21:  # VOC - Multi-class
            self.module = MultiClassModule(channels)
        elif num_classes == 2:  # Binary datasets
            self.module = BinarySpecializedModule(channels)
        else:
            self.module = AdaptiveModule(channels)
            
    def forward(self, x):
        return self.module(x)

class MultiClassModule(nn.Module):
    """VOC 다중클래스 특화 (Object boundaries + Context)"""
    def __init__(self, channels):
        super().__init__()
        # Multi-scale context for objects
        self.aspp = LightASPP(channels, channels, [1, 3, 6])
        self.class_attention = EfficientChannelAttention(channels)
        
    def forward(self, x):
        x = self.aspp(x)
        x = self.class_attention(x)
        return x

class BinarySpecializedModule(nn.Module):
    """Binary 데이터셋 특화 (Edge + Small region focus)"""
    def __init__(self, channels):
        super().__init__()
        # Edge enhancement for cracks/medical boundaries
        self.edge_enhance = EdgeEnhancement(channels)
        self.boundary_refine = BoundaryRefinement(channels)
        
    def forward(self, x):
        x = self.edge_enhance(x)
        x = self.boundary_refine(x)
        return x

class AdaptiveModule(nn.Module):
    """다른 클래스 수에 대한 adaptive 처리"""
    def __init__(self, channels):
        super().__init__()
        self.adaptive_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.adaptive_conv(x)))

# --- Efficient Building Blocks ---

class LightASPP(nn.Module):
    """경량화된 ASPP (Atrous Spatial Pyramid Pooling)"""
    def __init__(self, in_channels, out_channels, dilations=[1, 3, 6]):
        super().__init__()
        branch_channels = max(4, out_channels // len(dilations))
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.SiLU(inplace=True)
            ))
        
        self.fusion = nn.Conv2d(branch_channels * len(dilations), out_channels, 1, bias=False)
        
    def forward(self, x):
        features = []
        for branch in self.branches:
            features.append(branch(x))
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)

class EfficientChannelAttention(nn.Module):
    """ECA-Net 기반 채널 어텐션"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)  # B, C, 1, 1
        y = y.squeeze(-1).transpose(-1, -2)  # B, 1, C
        
        # 1D convolution
        y = self.conv(y)  # B, 1, C
        y = y.transpose(-1, -2).unsqueeze(-1)  # B, C, 1, 1
        
        # Attention weights
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class EdgeEnhancement(nn.Module):
    """균열/경계 강화 모듈"""
    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gradient_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)
        
    def forward(self, x):
        # Edge detection
        edge_feat = self.edge_conv(x)
        
        # Gradient computation (Sobel-like)
        grad_feat = self.gradient_conv(x)
        
        # Combine edge and gradient information
        combined = torch.cat([edge_feat, grad_feat], dim=1)
        enhanced = self.fusion(combined)
        
        return x + enhanced

class BoundaryRefinement(nn.Module):
    """경계 세부화 모듈"""
    def __init__(self, channels):
        super().__init__()
        self.boundary_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.refine_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        boundary = self.boundary_conv(x)
        refined = self.refine_conv(boundary)
        return x + refined

# --- Main Architecture ---

class submission_CartesianNet_Ultra(nn.Module):
    """혁신적 Cartesian Decomposition 기반 초경량 Segmentation 모델
    
    핵심 혁신:
    1. LeMoRe의 Cartesian decomposition (3 orthogonal views)
    2. Nested attention for implicit learning  
    3. Dataset-aware specialized modules
    4. Advanced loss function integration
    5. <10K parameters로 0.43+ IoU 목표
    """
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # Base channels (very conservative for <10K params)
        base_ch = 16
        
        # Initial projection
        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(base_ch)
        self.input_act = nn.SiLU(inplace=True)
        
        # Cartesian Encoder stages
        self.cart_enc1 = CartesianEncoder(base_ch, base_ch * 2)      # 16->32
        self.cart_enc2 = CartesianEncoder(base_ch * 2, base_ch * 3)  # 32->48
        
        # Nested Attention in bottleneck
        self.nested_attn = NestedAttention(base_ch * 3, reduction=3)
        
        # Dataset-aware processing
        self.dataset_module = DatasetAwareModule(base_ch * 3, num_classes)
        
        # Decoder with skip connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 3, base_ch * 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2 + base_ch * 2, base_ch, 2, stride=2, bias=False),  # Skip connection
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True)
        )
        
        # Final prediction
        self.final_conv = nn.Conv2d(base_ch + base_ch, num_classes, 1)  # Skip connection
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Input processing
        x1 = self.input_act(self.input_bn(self.input_conv(x)))  # B, 16, H, W
        
        # Cartesian encoding with downsampling
        x2 = F.max_pool2d(x1, 2)  # B, 16, H/2, W/2
        x2 = self.cart_enc1(x2)   # B, 32, H/2, W/2
        
        x3 = F.max_pool2d(x2, 2)  # B, 32, H/4, W/4
        x3 = self.cart_enc2(x3)   # B, 48, H/4, W/4
        
        # Nested attention in bottleneck
        x3 = self.nested_attn(x3)  # B, 48, H/4, W/4
        
        # Dataset-aware processing
        x3 = self.dataset_module(x3)  # B, 48, H/4, W/4
        
        # Decoder with skip connections
        d2 = self.dec2(x3)  # B, 32, H/2, W/2
        d2 = torch.cat([d2, x2], dim=1)  # B, 64, H/2, W/2
        
        d1 = self.dec1(d2)  # B, 16, H, W
        d1 = torch.cat([d1, x1], dim=1)  # B, 32, H, W
        
        # Final prediction
        out = self.final_conv(d1)  # B, num_classes, H, W
        
        return out


# --- Advanced Loss Functions ---

class AdaptiveFocalTverskyLoss(nn.Module):
    """데이터셋별 특화 Loss Function 통합"""
    def __init__(self, num_classes, alpha=0.3, beta=0.7, gamma=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # False positive penalty
        self.beta = beta    # False negative penalty  
        self.gamma = gamma  # Focal parameter
        
    def forward(self, inputs, targets):
        # Softmax for multi-class, Sigmoid for binary
        if self.num_classes > 2:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
            
        # Compute Tversky index
        smooth = 1e-6
        tversky_loss = 0
        
        for i in range(self.num_classes):
            if self.num_classes > 2:
                true_pos = (inputs[:, i] * targets[:, i]).sum()
                false_pos = (inputs[:, i] * (1 - targets[:, i])).sum() 
                false_neg = ((1 - inputs[:, i]) * targets[:, i]).sum()
            else:
                if i == 0:  # Background
                    pred, targ = 1 - inputs, 1 - targets
                else:  # Foreground
                    pred, targ = inputs, targets
                    
                true_pos = (pred * targ).sum()
                false_pos = (pred * (1 - targ)).sum()
                false_neg = ((1 - pred) * targ).sum()
            
            tversky = (true_pos + smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + smooth)
            
            # Focal weighting for hard examples
            focal_weight = (1 - tversky) ** self.gamma
            tversky_loss += focal_weight * (1 - tversky)
            
        return tversky_loss / self.num_classes


def test_model():
    """모델 테스트 및 파라미터 수 확인"""
    print("🧪 CartesianNet-Ultra 모델 테스트")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        (3, 2, "Binary (ETIS/CFD/CarDD)"),
        (3, 21, "Multi-class (VOC)")
    ]
    
    for in_ch, num_classes, desc in configs:
        model = submission_CartesianNet_Ultra(in_ch, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        test_input = torch.randn(1, in_ch, 256, 256)
        try:
            output = model(test_input)
            print(f"✅ {desc}")
            print(f"   파라미터: {total_params:,}개")
            print(f"   입력: {test_input.shape}")
            print(f"   출력: {output.shape}")
            print(f"   파라미터 제한: {'✅ 통과' if total_params <= 10000 else '❌ 초과'}")
            print()
        except Exception as e:
            print(f"❌ {desc}: {e}")
            print()

if __name__ == "__main__":
    test_model() 