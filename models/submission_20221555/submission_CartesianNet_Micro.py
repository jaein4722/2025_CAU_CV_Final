import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Ultra-Lightweight Building Blocks ---

class MicroCartesianView(nn.Module):
    """극경량 Cartesian View 추출"""
    def __init__(self, channels):
        super().__init__()
        # 단일 1x1 conv로 view 변환
        self.view_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        # Simple view transformation
        return self.view_conv(x)

class MicroFusion(nn.Module):
    """극경량 Multi-view Fusion"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 최소한의 fusion
        self.fusion = nn.Conv2d(in_channels * 3, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, views):
        fused = torch.cat(views, dim=1)
        out = self.fusion(fused)
        return self.act(self.bn(out))

class MicroAttention(nn.Module):
    """최소한의 어텐션 (SE 방식)"""
    def __init__(self, channels):
        super().__init__()
        reduced = max(1, channels // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        att = self.gap(x)
        att = F.relu(self.fc1(att))
        att = self.sigmoid(self.fc2(att))
        return x * att

class MicroDatasetModule(nn.Module):
    """데이터셋별 극경량 특화"""
    def __init__(self, channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        if num_classes == 21:  # VOC - multi-scale context
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False)
        else:  # Binary - edge focus
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
            
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.num_classes == 21:
            # Multi-scale for VOC
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            out = x1 + x2
        else:
            # Edge enhancement for binary
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            out = x1 + x2
            
        return self.act(self.bn(out))

# --- Main Micro Architecture ---

class submission_CartesianNet_Micro(nn.Module):
    """극경량 Cartesian-inspired Segmentation Model
    
    목표: <10K parameters로 0.43+ IoU 달성
    
    핵심 설계 원칙:
    1. Cartesian view 아이디어 유지하되 극도 경량화
    2. 필수 모듈만 보존 (어텐션, 데이터셋 특화)
    3. 파라미터 효율성 최우선
    """
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # 극도로 작은 채널 수
        base_ch = 8
        
        # 1. Input projection (최소한)
        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(base_ch)
        
        # 2. Micro Cartesian Views
        self.view1 = MicroCartesianView(base_ch)      # Transverse
        self.view2 = MicroCartesianView(base_ch)      # Frontal  
        self.view3 = MicroCartesianView(base_ch)      # Lateral
        
        # 3. Multi-view Fusion
        self.fusion1 = MicroFusion(base_ch, base_ch * 2)  # 8->16
        
        # 4. Downsampling encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 3, 3, stride=2, padding=1, bias=False),  # 16->24, H/2
            nn.BatchNorm2d(base_ch * 3),
            nn.ReLU(inplace=True)
        )
        
        # 5. Second Cartesian stage
        self.view4 = MicroCartesianView(base_ch * 3)
        self.view5 = MicroCartesianView(base_ch * 3)  
        self.view6 = MicroCartesianView(base_ch * 3)
        
        self.fusion2 = MicroFusion(base_ch * 3, base_ch * 4)  # 24->32
        
        # 6. Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 5, 3, stride=2, padding=1, bias=False),  # 32->40, H/4
            nn.BatchNorm2d(base_ch * 5),
            nn.ReLU(inplace=True)
        )
        
        # 7. Micro Attention
        self.attention = MicroAttention(base_ch * 5)
        
        # 8. Dataset-aware processing
        self.dataset_module = MicroDatasetModule(base_ch * 5, num_classes)
        
        # 9. Decoder (minimal)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 5, base_ch * 3, 2, stride=2, bias=False),  # 40->24, H/2
            nn.BatchNorm2d(base_ch * 3),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 3 + base_ch * 4, base_ch * 2, 2, stride=2, bias=False),  # (24+32)->16, H
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )
        
        # 10. Final prediction
        self.final_conv = nn.Conv2d(base_ch * 2 + base_ch * 2, num_classes, 1)  # (16+16)->classes
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # 1. Input processing
        x1 = F.relu(self.input_bn(self.input_conv(x)))  # B, 8, H, W
        
        # 2. First Cartesian decomposition
        v1 = self.view1(x1)  # Transverse
        v2 = self.view2(x1.transpose(-2, -1)).transpose(-2, -1)  # Frontal
        v3 = self.view3(x1.flip(-1)).flip(-1)  # Lateral
        
        x2 = self.fusion1([v1, v2, v3])  # B, 16, H, W
        
        # 3. Encoder stage 1
        x3 = self.enc1(x2)  # B, 24, H/2, W/2
        
        # 4. Second Cartesian decomposition 
        v4 = self.view4(x3)  # Transverse
        v5 = self.view5(x3.transpose(-2, -1)).transpose(-2, -1)  # Frontal
        v6 = self.view6(x3.flip(-1)).flip(-1)  # Lateral
        
        x4 = self.fusion2([v4, v5, v6])  # B, 32, H/2, W/2
        
        # 5. Bottleneck
        x5 = self.bottleneck(x4)  # B, 40, H/4, W/4
        
        # 6. Attention
        x5 = self.attention(x5)  # B, 40, H/4, W/4
        
        # 7. Dataset-aware processing
        x5 = self.dataset_module(x5)  # B, 40, H/4, W/4
        
        # 8. Decoder
        d2 = self.dec2(x5)  # B, 24, H/2, W/2
        d2 = torch.cat([d2, x4], dim=1)  # B, 56, H/2, W/2
        
        d1 = self.dec1(d2)  # B, 16, H, W
        d1 = torch.cat([d1, x2], dim=1)  # B, 32, H, W
        
        # 9. Final prediction
        out = self.final_conv(d1)  # B, num_classes, H, W
        
        return out


# --- Advanced Loss for Training ---

class AdaptiveTverskyLoss(nn.Module):
    """간소화된 적응형 Tversky Loss"""
    def __init__(self, num_classes, alpha=0.3, beta=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # FP penalty
        self.beta = beta    # FN penalty
        
    def forward(self, inputs, targets):
        if self.num_classes > 2:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
            
        smooth = 1e-6
        total_loss = 0
        
        for i in range(self.num_classes):
            if self.num_classes > 2:
                pred = inputs[:, i]
                targ = targets[:, i]
            else:
                if i == 0:
                    pred, targ = 1 - inputs.squeeze(1), 1 - targets.squeeze(1)
                else:
                    pred, targ = inputs.squeeze(1), targets.squeeze(1)
                    
            tp = (pred * targ).sum()
            fp = (pred * (1 - targ)).sum()
            fn = ((1 - pred) * targ).sum()
            
            tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
            total_loss += 1 - tversky
            
        return total_loss / self.num_classes


def test_micro_model():
    """마이크로 모델 테스트"""
    print("🧪 CartesianNet-Micro 모델 테스트")
    print("=" * 50)
    
    configs = [
        (3, 2, "Binary (ETIS/CFD/CarDD)"),
        (3, 21, "Multi-class (VOC)")
    ]
    
    for in_ch, num_classes, desc in configs:
        model = submission_CartesianNet_Micro(in_ch, num_classes)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Forward test
        test_input = torch.randn(1, in_ch, 256, 256)
        try:
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✅ {desc}")
            print(f"   파라미터: {total_params:,}개")
            print(f"   입력: {test_input.shape}")
            print(f"   출력: {output.shape}")
            print(f"   파라미터 제한: {'✅ 통과' if total_params <= 10000 else '❌ 초과'}")
            print(f"   여유 파라미터: {10000 - total_params:,}개")
            print()
        except Exception as e:
            print(f"❌ {desc}: {e}")
            print()

if __name__ == "__main__":
    test_micro_model() 