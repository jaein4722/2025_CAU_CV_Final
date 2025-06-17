#!/usr/bin/env python3
"""
재현성 테스트 스크립트
동일한 시드로 여러 번 실행했을 때 결과가 동일한지 확인
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
import importlib

# 프로젝트 모듈 import
from competition_utils import control_random_seed, ImagesDataset, SegDataLoader
from training_args import Make_Optimizer, Make_LR_Scheduler, Make_Loss_Function

def test_basic_reproducibility():
    """기본 재현성 테스트 - 동일한 연산을 여러 번 수행"""
    print("🧪 기본 재현성 테스트 시작")
    
    seed = 42
    results = []
    
    for i in range(3):
        print(f"\n--- 테스트 {i+1}/3 ---")
        control_random_seed(seed)
        
        # 간단한 텐서 연산
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.mm(x, y)
        result = z.sum().item()
        
        print(f"  결과: {result:.8f}")
        results.append(result)
    
    # 결과 비교
    all_same = all(abs(r - results[0]) < 1e-10 for r in results)
    print(f"\n✅ 기본 재현성: {'통과' if all_same else '실패'}")
    if not all_same:
        print(f"  결과들: {results}")
    
    return all_same

def test_model_reproducibility():
    """모델 초기화 재현성 테스트"""
    print("\n🧪 모델 초기화 재현성 테스트 시작")
    
    # 간단한 테스트 모델
    class SimpleModel(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, num_classes, 3, padding=1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    seed = 42
    model_weights = []
    
    for i in range(3):
        print(f"\n--- 모델 테스트 {i+1}/3 ---")
        control_random_seed(seed)
        
        model = SimpleModel(3, 2)
        weights = model.conv1.weight.data.clone()
        weight_sum = weights.sum().item()
        
        print(f"  Conv1 가중치 합: {weight_sum:.8f}")
        model_weights.append(weight_sum)
    
    # 결과 비교
    all_same = all(abs(w - model_weights[0]) < 1e-10 for w in model_weights)
    print(f"\n✅ 모델 초기화 재현성: {'통과' if all_same else '실패'}")
    if not all_same:
        print(f"  가중치 합들: {model_weights}")
    
    return all_same

def test_dataloader_reproducibility():
    """DataLoader 재현성 테스트"""
    print("\n🧪 DataLoader 재현성 테스트 시작")
    
    # 더미 데이터셋 생성
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # 인덱스 기반으로 결정적 데이터 생성
            np.random.seed(idx)  # 각 샘플마다 고정 시드
            image = torch.randn(3, 32, 32)
            mask = torch.randint(0, 2, (32, 32))
            return image, mask, f"dummy_{idx}.jpg"
    
    seed = 42
    batch_orders = []
    
    for i in range(3):
        print(f"\n--- DataLoader 테스트 {i+1}/3 ---")
        control_random_seed(seed)
        
        dataset = DummyDataset(20)
        dataloader = SegDataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True,  # 셔플 활성화
            num_workers=2,  # 멀티워커 사용
            drop_last=False
        )
        
        batch_order = []
        for batch_idx, (images, masks, paths) in enumerate(dataloader):
            # 첫 번째 이미지의 첫 번째 픽셀 값으로 배치 식별
            identifier = images[0, 0, 0, 0].item()
            batch_order.append(round(identifier, 6))
            if batch_idx >= 2:  # 처음 3개 배치만 확인
                break
        
        print(f"  배치 순서: {batch_order}")
        batch_orders.append(batch_order)
    
    # 결과 비교
    all_same = all(batch_orders[0] == order for order in batch_orders)
    print(f"\n✅ DataLoader 재현성: {'통과' if all_same else '실패'}")
    if not all_same:
        print(f"  배치 순서들이 다름:")
        for i, order in enumerate(batch_orders):
            print(f"    테스트 {i+1}: {order}")
    
    return all_same

def test_environment_info():
    """환경 정보 출력"""
    print("\n📋 환경 정보:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 개수: {torch.cuda.device_count()}")
        print(f"  현재 GPU: {torch.cuda.current_device()}")
        print(f"  GPU 이름: {torch.cuda.get_device_name()}")
    
    print(f"  PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', '설정되지 않음')}")
    print(f"  CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', '설정되지 않음')}")

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("🔬 재현성 테스트 시작")
    print("=" * 60)
    
    test_environment_info()
    
    # 각 테스트 실행
    tests = [
        ("기본 연산", test_basic_reproducibility),
        ("모델 초기화", test_model_reproducibility),
        ("DataLoader", test_dataloader_reproducibility),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} 테스트 중 오류 발생: {e}")
            results[test_name] = False
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 최종 결과:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 전체 재현성: {'✅ 완벽' if all_passed else '❌ 문제 있음'}")
    
    if not all_passed:
        print("\n💡 재현성 문제 해결 방법:")
        print("  1. 프로그램 재시작 후 다시 테스트")
        print("  2. num_workers=0으로 설정하여 단일 스레드 사용")
        print("  3. 다른 컴퓨터에서는 동일한 PyTorch/CUDA 버전 사용")
        print("  4. control_random_seed(seed, use_deterministic=False) 시도")

if __name__ == "__main__":
    main() 