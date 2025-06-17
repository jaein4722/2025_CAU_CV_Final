"""
VOC 파이프라인 점검 스크립트
──────────────────────────
1. Dataset 로드 (이미지·마스크)
2. 마스크 라벨 무결성 검사 (0–20, 255)
3. ignore_index 픽셀 비율 출력
4. IoU 함수 Sanity Check (GT=Pred → 1.0, Rand → ≈0)
5. 리사이즈 보간 테스트 (새 라벨 생성 여부)
6. 모델 출력 채널·파라미터 업데이트 테스트
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import random

# ──────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────────────────
class VOCDataset(Dataset):
    def __init__(self, split="train", root="Datasets/VOC", to_tensor=True):
        self.root = Path(root) / split
        self.img_paths = sorted((self.root / "Originals").glob("*"))
        self.mask_paths = sorted((self.root / "Masks").glob("*.npy"))
        assert len(self.img_paths) == len(self.mask_paths), "이미지·마스크 개수 불일치"
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = read_image(str(img_path))  # (C,H,W) uint8 [0,255]
        mask = torch.from_numpy(np.load(mask_path)).long()  # (H,W) int64
        if self.to_tensor:
            img = img.float() / 255.0
        return img, mask

# ──────────────────────────────────────────────────────────────────────────
# 2. 유틸리티: IoU (mean over classes 0–20, 배경 포함)
# ──────────────────────────────────────────────────────────────────────────
def mean_iou(pred, target, num_classes=21, ignore_index=255, eps=1e-6):
    pred = pred.clone()
    target = target.clone()
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        if target_inds.sum() == 0:  # GT에 없는 클래스는 스킵
            continue
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        ious.append((intersection + eps) / (union + eps))
    return np.mean(ious) if ious else 0.0

# ──────────────────────────────────────────────────────────────────────────
# 3. 메인 검사 루틴
# ──────────────────────────────────────────────────────────────────────────
def main():
    ds = VOCDataset(split="train")
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    print(f"[INFO] Train 샘플 수: {len(ds)}")

    # (A) 라벨·ignore_index 확인 (최대 100장 샘플링)
    uniq_global = set()
    ignore_count = 0
    total_px = 0
    for i, (_, mask) in enumerate(loader):
        uniq_global.update(mask.unique().tolist())
        ignore_count += (mask == 255).sum().item()
        total_px += mask.numel()
        if i == 99:  # 100장만 확인
            break
    print(f"[CHECK] 마스크 전체 유니크 값: {sorted(uniq_global)}")
    print(f"[CHECK] ignore_index(255) 픽셀 비율: {ignore_count/total_px*100:.3f} %")

    # (B) IoU 함수 Sanity Check
    img, mask = ds[0]
    gt_pred = mask.clone()
    random_pred = torch.randint(0, 21, mask.shape)
    print(f"[SANITY] GT==Pred IoU: {mean_iou(gt_pred, mask):.3f} (→ 1.0이어야)")
    print(f"[SANITY] Random IoU:  {mean_iou(random_pred, mask):.3f} (≈ 0 기대)")

    # (C) 보간 테스트: nearest vs bilinear
    h, w = mask.shape
    mask_nearest = resize(mask.unsqueeze(0).float(), size=[h//2, w//2],
                          interpolation=InterpolationMode.NEAREST).squeeze().long()
    mask_bilinear = resize(mask.unsqueeze(0).float(), size=[h//2, w//2],
                           interpolation=InterpolationMode.BILINEAR).squeeze().long()
    uniq_nearest = torch.unique(mask_nearest)
    uniq_bilinear = torch.unique(mask_bilinear)
    print(f"[RESIZE] NEAREST 라벨 집합 : {uniq_nearest.tolist()}")
    print(f"[RESIZE] BILINEAR 라벨 집합: {uniq_bilinear.tolist()} (➡ 새 값이 생기면 오류)")

    # (D) 모델 출력·파라미터 업데이트 체크
    try:
        from models.submission_20221555.submission_UNet_basic import submission_UNet_basic  # 프로젝트 구조에 맞게 경로 조정
        net = submission_UNet_basic(in_channels=3, num_classes=21)
    except ImportError:
        # 더미 네트워크 (Conv 3→21)
        class DummyNet(nn.Module):
            def __init__(self): super().__init__(); self.outc=nn.Conv2d(3,21,1)
            def forward(self,x): return self.outc(x)
        net = DummyNet()

    x = img.unsqueeze(0)            # (1,3,H,W)
    y = net(x)                      # (1,21,H,W) ?
    print(f"[MODEL] 출력 shape : {tuple(y.shape)}  (→ (1,21,H,W)?)")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    before = net.outc.weight.clone()
    loss = criterion(y, mask.unsqueeze(0))
    loss.backward(); optimizer.step()
    updated = not torch.equal(before, net.outc.weight)
    print(f"[MODEL] 파라미터 업데이트 발생 여부: {updated}")

if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    main()
