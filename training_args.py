import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def Make_Optimizer(model):
    magic = "sgd"
    
    if magic == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
    elif magic == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        
    elif magic == "sgd_lcnet":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
        
    elif magic == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        
    elif magic == "baseline":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    else:
        raise ValueError(f"Unsupported optimizer: {magic}. Use 'adam' or 'sgd'.")
    return optimizer


def Make_LR_Scheduler(optimizer):
    magic = "warmup_cosine"
    
    if magic == "warmup_cosine":
        lr_scheduler = WarmupCosineLR(optimizer, T_max = 30, warmup_iters = 2, eta_min = 1e-6)
        
    elif magic == "warmup_poly":
        lr_scheduler = WarmupPolyLR(optimizer, T_max = 30, warmup_iters = 2, eta_min = 1e-6, power = 0.9)
        
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
        loss = CrossEntropyLoss2d(mode=WORK_MODE)
    else:
        loss = DiceCELoss(mode=WORK_MODE)
    
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
    Polynomial decay + linear warm-up (epoch 단위 스케줄)
    """
    def __init__(
        self,
        optimizer,
        T_max: int,
        cur_iter: int = 0,
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 500,
        eta_min: float = 0.0,
        power: float = 0.9,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters  = warmup_iters
        self.power         = power
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_iter - 1)

    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs

        # 1) Warm-up
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / float(max(1, self.warmup_iters))
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        # 2) Polynomial decay
        progress = (self.last_epoch - self.warmup_iters) / float(
            max(1, self.T_max - self.warmup_iters)
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 - progress) ** self.power
            for base_lr in self.base_lrs
        ]


class FocalLoss2d(nn.Module):
    '''
    Focal Loss supporting both binary and multi-class segmentation
    '''
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, ignore_index=255, mode='binary'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.mode = mode

    def forward(self, preds, targets):
        if self.mode == 'binary':  # Binary segmentation case (B, 1, H, W)
            preds = preds.squeeze(1)  # (B, H, W)
            preds = torch.sigmoid(preds)
            targets = targets.squeeze(1).float() if targets.dim() > 3 else targets.float()

            pt = preds * targets + (1 - preds) * (1 - targets)
            logpt = torch.log(pt + 1e-9)
            loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt
            return loss.mean()
        elif self.mode == 'multiclass':  # Multi-class case (B, C, H, W)
            log_probs = F.log_softmax(preds, dim=1)  # (B, C, H, W)
            probs = torch.exp(log_probs)
            # targets가 (B, 1, H, W) 형태라면 (B, H, W)로 squeeze
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1])  # (B, H, W, C)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()     # (B, C, H, W)

            pt = (probs * targets_one_hot).sum(1)  # (B, H, W)
            logpt = (log_probs * targets_one_hot).sum(1)  # (B, H, W)
            loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt

            if self.ignore_index is not None:
                mask = targets != self.ignore_index
                return loss[mask].mean()
            else:
                return loss.mean()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'binary' or 'multiclass'.")
        
        
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, mode='multiclass'):
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.weight = weight

        if self.mode == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)  # weight 대신 pos_weight 사용 가능
        elif self.mode == 'multiclass':
            self.loss_fn = nn.NLLLoss(weight=weight, ignore_index=ignore_index)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Use 'binary' or 'multiclass'.")

    def forward(self, outputs, targets):
        if self.mode == 'binary':
            # outputs: (B, 1, H, W), targets: (B, 1, H, W) or (B, H, W)
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1).float()
            outputs = outputs.squeeze(1)
            return self.loss_fn(outputs, targets)
        elif self.mode == 'multiclass':
            # outputs: (B, C, H, W), targets: (B, H, W)
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            return self.loss_fn(F.log_softmax(outputs, dim=1), targets)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'.")


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


class DynamicSegLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, eps: float = 1e-6, mode='multiclass'):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.mode = mode

    # ---------- 공통 유틸 ----------
    def _flatten(self, probs, targets):
        # (N, C, H, W) -> (C, N*H*W) with ignore mask
        probs = probs.permute(1, 0, 2, 3).reshape(probs.shape[1], -1)
        targets = targets.view(-1)
        valid = targets != self.ignore_index
        return probs[:, valid], targets[valid]

    # ---------- Lovász ----------
    @torch.no_grad()
    def _lovasz_grad(self, gt_sorted):
        gtsum = gt_sorted.sum()
        if gtsum == 0:
            return gt_sorted.new_ones(gt_sorted.size())
        inter = gtsum - gt_sorted.cumsum(0)
        union = gtsum + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - inter / union
        return torch.cat((jaccard[0:1], jaccard[1:] - jaccard[:-1]))

    def _lovasz_softmax(self, probs, targets):
        # probs: (C, P) softmax probs; targets: (P,) ints in [0,C-1]
        losses = []
        C = probs.shape[0]
        for c in range(C):
            fg = (targets == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        return torch.mean(torch.stack(losses))

    def _lovasz_hinge(self, logits, targets):
        # logits: (P,) raw; targets: (P,) {0,1}
        signs = 2. * targets.float() - 1.
        errors = 1. - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        targets_sorted = targets[perm]
        grad = self._lovasz_grad(targets_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    # ---------- Generalized Dice ----------
    def _gen_dice(self, probs, targets_1hot):
        dims = (1,) if probs.ndim == 2 else tuple(range(2, probs.ndim))
        w = 1. / (targets_1hot.sum(dims, keepdim=True) + self.eps) ** 2
        inter = (w * probs * targets_1hot).sum(dims)
        denom = (w * (probs + targets_1hot)).sum(dims)
        return 1. - (2 * inter + self.eps) / (denom + self.eps)

    # ---------- Dynamic Tversky ----------
    def _tversky(self, probs, targets):
        dims = tuple(range(2, targets.ndim))
        p = targets.sum() / (targets.numel() + self.eps)  # foreground ratio
        alpha, beta = p, 1 - p
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        return (tp + self.eps) / (tp + alpha * fp + beta * fn + self.eps)

    # ---------- forward ----------
    def forward(self, logits, targets):
        N, C, H, W = logits.shape
        if self.mode == 'binary':
            # -------- Binary branch --------
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction='mean')
            probs = torch.sigmoid(logits)
            # Lovász-Hinge (per image to avoid memory blow-up)
            lav = torch.mean(torch.stack([
                self._lovasz_hinge(
                    probs[i, 0].flatten(), targets[i, 0].flatten())
                for i in range(N)
            ]))
            # Dynamic Tversky (1 - Tversky → loss)
            tv = 1 - self._tversky(probs, targets.float()).mean()
            return bce + lav + tv

        elif self.mode == 'multiclass':
            # -------- Multiclass branch --------
            targets = targets.squeeze(1)
            ce = F.cross_entropy(logits, targets.long(),
                                 ignore_index=self.ignore_index)
            probs = F.softmax(logits, dim=1)
            # Lovász-Softmax
            p_flat, t_flat = self._flatten(probs, targets)
            lav = self._lovasz_softmax(p_flat, t_flat)
            # Generalized Dice
            t_onehot = F.one_hot(
                torch.clamp(targets, 0), num_classes=C
            ).permute(0, 3, 1, 2).float()
            gd = self._gen_dice(probs, t_onehot).mean()
            return ce + lav + gd
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'binary' or 'multiclass'.")