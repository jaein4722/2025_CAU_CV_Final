# 🔄 재현성 문제 해결 가이드

## 📋 문제 상황
시드 고정 메커니즘이 존재함에도 불구하고:
- 동일한 컴퓨터에서 실험을 새로 할 때마다 결과가 다름
- 다른 컴퓨터에서 실행할 때 결과가 전혀 재현되지 않음

## 🔍 원인 분석

### 1. **DataLoader 멀티스레딩 이슈** ⚠️
```python
# 문제 코드
train_loader = SegDataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
```
- `num_workers=4`로 멀티스레딩 사용
- `worker_init_fn`이 설정되지 않아 각 워커가 다른 시드 사용
- 배치 순서가 매번 달라짐

### 2. **환경변수 미설정**
```python
# 누락된 설정들
os.environ['PYTHONHASHSEED'] = str(seed)  # Python 해시 함수 결정성
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA 연산 동기화
```

### 3. **CUDA 비결정성**
- `torch.backends.cudnn.deterministic = True`만으로는 부족
- `torch.use_deterministic_algorithms(True)` 필요 (PyTorch 1.8+)

### 4. **하드웨어/환경 차이**
- GPU 아키텍처 차이 (RTX 3080 vs RTX 4090)
- PyTorch/CUDA/cuDNN 버전 차이
- 드라이버 버전 차이

## ✅ 해결 방법

### 1. **강화된 시드 설정 함수 사용**
```python
# 이미 수정된 competition_utils.py의 control_random_seed 함수 사용
control_random_seed(seed=42, use_deterministic=True)
```

### 2. **DataLoader 설정 확인**
```python
# 자동으로 worker_init_fn이 설정되도록 수정됨
train_loader = SegDataLoader(
    train_dataset, 
    batch_size=16, 
    num_workers=4,  # worker_init_fn 자동 설정됨
    shuffle=True
)
```

### 3. **재현성 테스트 실행**
```bash
python test_reproducibility.py
```

### 4. **문제 발생 시 대안**

#### 옵션 1: 단일 스레드 사용 (가장 확실)
```python
# num_workers=0으로 설정
train_loader = SegDataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)
```

#### 옵션 2: 결정적 알고리즘 비활성화
```python
# 성능 우선, 부분적 재현성
control_random_seed(seed=42, use_deterministic=False)
```

#### 옵션 3: 환경 통일
```bash
# 동일한 환경 구성
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🧪 재현성 검증 방법

### 1. **빠른 테스트**
```python
# 동일한 모델을 3번 생성하여 가중치 비교
for i in range(3):
    control_random_seed(42)
    model = YourModel(3, 21)
    print(f"가중치 합: {sum(p.sum().item() for p in model.parameters())}")
```

### 2. **전체 테스트**
```bash
python test_reproducibility.py
```

### 3. **실제 실험 테스트**
```python
# 동일한 설정으로 2번 실행하여 결과 비교
python competition_main.ipynb  # 첫 번째 실행
# results.csv 백업
python competition_main.ipynb  # 두 번째 실행
# 결과 비교
```

## 📊 재현성 수준별 대응

### Level 1: 완전 재현성 (권장)
```python
control_random_seed(42, use_deterministic=True)
# num_workers=0 또는 worker_init_fn 사용
# 동일한 하드웨어/소프트웨어 환경
```
- **장점**: 100% 재현 가능
- **단점**: 훈련 속도 약간 느림

### Level 2: 부분 재현성
```python
control_random_seed(42, use_deterministic=False)
# num_workers > 0, worker_init_fn 사용
```
- **장점**: 빠른 훈련 속도
- **단점**: 미세한 차이 발생 가능

### Level 3: 통계적 재현성
```python
# 여러 번 실행하여 평균값 사용
results = []
for seed in [42, 43, 44, 45, 46]:
    control_random_seed(seed)
    result = run_experiment()
    results.append(result)
mean_result = np.mean(results)
```

## 🚨 주의사항

### 1. **성능 vs 재현성 트레이드오프**
- `torch.use_deterministic_algorithms(True)`: 완전 재현성, 느린 속도
- `torch.backends.cudnn.benchmark = False`: 재현성 우선, 속도 저하

### 2. **하드웨어별 차이**
```python
# GPU별 결과가 다를 수 있는 연산들
- torch.addmm (행렬 곱셈)
- torch.bmm (배치 행렬 곱셈)  
- torch.conv2d (합성곱)
- torch.nn.functional.interpolate (보간)
```

### 3. **버전별 차이**
- PyTorch 1.7 vs 1.8+: `use_deterministic_algorithms` 지원 여부
- CUDA 10.2 vs 11.x: 일부 연산 결과 차이
- cuDNN 7.x vs 8.x: 합성곱 알고리즘 차이

## 🔧 문제 해결 체크리스트

### 실험 전 확인사항
- [ ] `control_random_seed(seed)` 호출 확인
- [ ] `worker_init_fn` 설정 확인 (num_workers > 0인 경우)
- [ ] 환경변수 설정 확인 (`PYTHONHASHSEED`, `CUDA_LAUNCH_BLOCKING`)
- [ ] PyTorch/CUDA 버전 확인

### 문제 발생 시 단계별 해결
1. **재현성 테스트 실행**: `python test_reproducibility.py`
2. **단일 스레드 시도**: `num_workers=0`
3. **결정적 알고리즘 비활성화**: `use_deterministic=False`
4. **환경 정보 확인**: GPU, PyTorch 버전
5. **통계적 접근**: 여러 시드로 실험 후 평균

### 다른 컴퓨터에서 재현 시
1. **동일한 환경 구성**: requirements.txt 사용
2. **GPU 정보 확인**: `nvidia-smi`, `torch.cuda.get_device_name()`
3. **라이브러리 버전 통일**: `pip freeze > requirements.txt`
4. **데이터 무결성 확인**: 데이터셋 해시값 비교

## 📈 성능 영향 분석

| 설정 | 재현성 | 훈련 속도 | 권장 상황 |
|------|--------|-----------|-----------|
| `use_deterministic=True, num_workers=0` | 100% | 느림 | 최종 실험, 논문 제출 |
| `use_deterministic=True, num_workers>0` | 95% | 보통 | 일반적 실험 |
| `use_deterministic=False, num_workers>0` | 90% | 빠름 | 빠른 프로토타이핑 |
| 기본 설정 | 70% | 가장 빠름 | 초기 탐색 |

## 💡 권장 워크플로우

### 개발 단계
```python
# 빠른 실험을 위한 설정
control_random_seed(42, use_deterministic=False)
num_workers = 4  # 빠른 데이터 로딩
```

### 검증 단계  
```python
# 재현성 확인을 위한 설정
control_random_seed(42, use_deterministic=True)
num_workers = 0  # 완전한 재현성
```

### 최종 제출
```python
# 완전한 재현성 보장
control_random_seed(42, use_deterministic=True)
num_workers = 0
# 여러 시드로 실험하여 안정성 확인
seeds = [42, 43, 44, 45, 46]
``` 