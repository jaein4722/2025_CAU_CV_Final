import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def Make_Optimizer(model):
    magic = "adamw"
    
    if magic == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    elif magic == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        
    elif magic == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    
    elif magic == "baseline":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    else:
        raise ValueError(f"Unsupported optimizer: {magic}. Use 'adam' or 'sgd'.")
    return optimizer


def Make_LR_Scheduler(optimizer):
    magic = "warmup_poly"
    
    if magic == "warmup_cosine":
        lr_scheduler = WarmupCosineLR(optimizer, T_max = 30, warmup_iters = 2, eta_min = 1e-6)
        
    elif magic == "warmup_poly":
        lr_scheduler = WarmupPolyLR(
            optimizer,
            T_max=30,               # 총 Epoch 수
            warmup_epochs=0,        # 5 Epoch warm-up
            warmup_factor=1/10,     # 시작 lr = base_lr * 0.1
            power=0.9,
            eta_min=1e-6)
        
    elif magic == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        
    elif magic == "baseline":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min = 1e-6)
        
    else:
        raise ValueError(f"Unsupported lr scheduler: {magic}. Use 'cosine' or 'constant'.")
    return lr_scheduler


def Make_Loss_Function(number_of_classes):
    
    BINARY_SEG = True if number_of_classes==2 else False
    WORK_MODE = "binary" if BINARY_SEG else "multiclass"
    
    if BINARY_SEG:
        loss = DiceCELoss(mode=WORK_MODE)
    else:
        loss = UniformCBCE_Lovasz(number_of_classes)
    
    return loss

class WarmupCosineLR(_LRScheduler):
    """
    Cosine annealing + linear warm-up (epoch 단위 스케줄)
    """
    def __init__(
        self,
        optimizer,
        T_max: int,
        cur_iter: int = 0,              # 외부 호환용. epoch 기준이면 0, iter 기준이면 현재 step 수
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 500,
        eta_min: float = 0.0,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters  = warmup_iters
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_iter - 1)

    def get_lr(self):
        # 첫 호출( last_epoch == -1 ) → base_lr 그대로
        if self.last_epoch == -1:
            return self.base_lrs

        # 1) Warm-up 구간
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / float(max(1, self.warmup_iters))
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        # 2) Cosine annealing 구간
        progress = (self.last_epoch - self.warmup_iters) / float(
            max(1, self.T_max - self.warmup_iters)
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


class WarmupPolyLR(_LRScheduler):
    """
    Linear warm-up → Polynomial decay  (epoch 단위 step 전용)

        lr = base_lr * { w + (1-w)*e/E          (e < warmup_epochs)
                       { (1 - (e-w)/ (E-w) )^p  (else)

    Args
    ----
    optimizer      : torch.optim.* 인스턴스
    T_max (int)    : 학습 총 Epoch 수 (=E)
    warmup_epochs  : 워밍업 Epoch 수 (=w)
    warmup_factor  : 워밍업 시작 배율 (0.0~1.0). 0.1이면 base_lr×0.1에서 시작
    power (float)  : 멱 지수 p (보통 0.9)
    eta_min (float): 최종 최소 lr
    """
    def __init__(self,
                 optimizer,
                 T_max: int,
                 cur_epoch: int = 0,
                 warmup_epochs: int = 5,
                 warmup_factor: float = 1.0 / 10,
                 power: float = 0.9,
                 eta_min: float = 0.0):
        self.T_max         = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.power         = power
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_epoch - 1)  # _LRScheduler 내부에서 +1 됨

    # -------------------------------------------------
    def get_lr(self):
        # 첫 step 이전: base_lr 그대로
        if self.last_epoch == -1:
            return self.base_lrs

        # ---------- 1) Warm-up ----------
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / float(max(1, self.warmup_epochs))
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        # ---------- 2) Poly Decay ----------
        e = self.last_epoch - self.warmup_epochs
        E = self.T_max      - self.warmup_epochs
        poly = (1 - e / float(max(1, E))) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * poly
            for base_lr in self.base_lrs
        ]


class DiceCELoss:
    def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
        self.weight = weight
        self.epsilon = epsilon
        self.mode = mode
    
    def __call__(self, pred, target):
        if self.mode == 'binary':
            pred = pred.squeeze(1)  # shape: (batchsize, H, W)
            target = target.squeeze(1).float()
            intersection = torch.sum(pred * target, dim=(1, 2))
            union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.binary_cross_entropy(pred, target)
        
        elif self.mode == 'multiclass':
            batchsize, num_classes, H, W = pred.shape
            target = target.squeeze(1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
            intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
            union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.cross_entropy(pred, target)
        else:
            raise ValueError("mode should be 'binary' or 'multiclass'")
        
        combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
        
        return combined_loss


class UniformCBCE_Lovasz(nn.Module):
    """
    · 모든 전경 클래스 가중치는 1.0
    · 배경(class 0) 은 `bg_factor` (<1) 로 낮춰서 편향 완화
    · CE + Lovász-Softmax 결합 손실
    """
    def __init__(self,
                 num_classes: int,
                 bg_factor: float = 0.01,
                 coef_ce: float = 0.6,
                 coef_iou: float = 0.4):
        super().__init__()
        weight = torch.ones(num_classes, dtype=torch.float32)
        weight[0] = bg_factor                # 배경만 낮춤
        self.ce   = nn.CrossEntropyLoss(weight=weight)
        self.coeff_ce, self.coeff_iou = coef_ce, coef_iou

    # -------- Lovász 순수 PyTorch 구현 (간략 버전) --------
    @staticmethod
    def lovasz_softmax(probs, labels, classes='present'):
        # probs: (B,C,H,W), labels: (B,H,W)
        B, C, *_ = probs.shape
        losses = []
        for c in range(C):
            fg = (labels == c).float()                 # foreground mask
            if classes == 'present' and fg.sum() == 0:
                continue
            pc = probs[:, c, ...]
            errors = (fg - pc).abs()
            errors_sorted, perm = torch.sort(errors.view(B, -1), dim=1, descending=True)
            fg_sorted = fg.view(B, -1).gather(1, perm)
            grad = torch.cumsum(fg_sorted, dim=1) / fg_sorted.sum(dim=1, keepdim=True).clamp(min=1)
            loss = (errors_sorted * grad).sum(dim=1)   # (B,)
            # 🔧 픽셀 수로 정규화
            pixel_cnt = fg.numel()                     # H*W
            loss = (loss / pixel_cnt).mean()
            
            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=probs.device)

    def forward(self, logits, target):
        target = target.squeeze(1).long()         # (B,H,W)
        
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
            
        loss_ce  = self.ce(logits, target)
        loss_iou = self.lovasz_softmax(F.softmax(logits,1), target)
        return self.coeff_ce * loss_ce + self.coeff_iou * loss_iou