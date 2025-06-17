# 🔬 Optuna 하이퍼파라미터 최적화 실험 가이드

> **목적**: competition ipynb의 구조를 그대로 유지하면서 optuna를 통한 하이퍼파라미터 최적화 수행  
> **업데이트**: 2025년 6월 16일  
> **기반**: competition_modified.ipynb → optuna_experiment.py 변환

---

## 📋 목차

1. [시스템 개요](#-시스템-개요)
2. [파일 구조](#-파일-구조)
3. [실행 방법](#-실행-방법)
4. [설정 변경](#-설정-변경)
5. [결과 확인](#-결과-확인)
6. [문제 해결](#-문제-해결)
7. [성능 최적화 팁](#-성능-최적화-팁)

---

## 🏗️ 시스템 개요

### 주요 특징
- **기존 로직 보존**: competition_modified.ipynb의 실험 로직을 그대로 py 파일로 변환
- **백그라운드 실행**: nohup을 통한 장시간 실험 지원
- **스마트 Pruning**: 초기 성능이 너무 낮은 trial 자동 중단
- **결과 추적**: SQLite DB와 CSV 파일로 모든 실험 기록 저장

### 최적화 대상 하이퍼파라미터
```
✅ Optimizer: adam, adamw, sgd
✅ Learning Rate: 1e-4 ~ 1e-2 (log scale)
✅ Weight Decay: 1e-5 ~ 1e-3 (log scale)  
✅ Scheduler: warmup_cosine, warmup_poly, constant
✅ Loss Weight: DiceCE 손실의 dice/ce 비율 (0.3~0.7)
✅ Warmup Epochs: 0~10
```

---

## 📁 파일 구조

```
CV_final/
├── optuna_experiment.py          # 메인 실험 스크립트
├── optuna_config.py              # 설정 파일  
├── run_optuna.sh                 # 백그라운드 실행 스크립트
├── OPTUNA_사용법_가이드.md       # 이 파일
├── competition_utils.py          # 기존 유틸리티 (그대로 사용)
├── training_args.py              # 기존 훈련 설정 (참고용)
├── requirements.txt              # 업데이트된 의존성
└── logs/                         # 실험 로그 디렉토리
    ├── optuna_YYMMDD_HHMMSS.log
    └── optuna_YYMMDD_HHMMSS.pid
```

---

## 🚀 실행 방법

### 1. 의존성 설치
```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 또는 optuna만 별도 설치
pip install optuna>=3.0.0
```

### 2. 설정 확인
```bash
# 설정 파일 유효성 검사
python optuna_config.py
```

### 3. 실험 실행

#### 방법 A: 백그라운드 실행 (권장)
```bash
# 실행 권한 부여 (최초 1회)
chmod +x run_optuna.sh

# 백그라운드에서 실험 시작
./run_optuna.sh
```

#### 방법 B: 직접 실행
```bash
# 포그라운드에서 실행 (터미널 유지 필요)
python optuna_experiment.py

# 또는 백그라운드
nohup python optuna_experiment.py > logs/manual_run.log 2>&1 &
```

### 🔄 지속적 실험 모드 (NEW!)

**같은 데이터베이스를 계속 사용하여 실험 결과를 누적할 수 있습니다!**

#### 설정 방법
`optuna_config.py`에서 다음 설정 확인:
```python
STUDY_CONFIG = {
    # ... 기존 설정들 ...
    'persistent_study': True,              # 지속적 스터디 활성화
    'study_base_name': 'semantic_seg_opt', # 고정 스터디 이름
    'reset_study_on_start': False,         # 기존 데이터 유지
}
```

#### 사용법
```bash
# 첫 번째 실험 (10개 trial)
python optuna_experiment.py

# 두 번째 실험 (추가 10개 trial) - 같은 DB 사용
python optuna_experiment.py

# 세 번째 실험 (추가 10개 trial) - 계속 누적
python optuna_experiment.py
```

#### 지속적 모드 vs 일회성 모드
| 설정 | persistent_study=True | persistent_study=False |
|------|----------------------|------------------------|
| DB 파일명 | `semantic_seg_opt.db` (고정) | `semantic_segmentation_optimization_YYMMDD_HHMMSS.db` |
| 실험 누적 | ✅ 계속 누적 | ❌ 매번 새로 시작 |
| 이전 결과 | ✅ 보존 및 활용 | ❌ 새로 시작 |
| 최적 파라미터 | ✅ 전체 히스토리에서 선택 | ❌ 현재 세션만 |

#### 기존 스터디 초기화
```python
# 새로 시작하고 싶은 경우
STUDY_CONFIG = {
    'persistent_study': True,
    'reset_study_on_start': True,  # ⚠️ 기존 데이터 삭제!
    # ...
}
```

---

## ⚙️ 설정 변경

### 기본 실험 설정 수정
`optuna_config.py` 파일에서 다음 설정들을 변경할 수 있습니다:

```python
# 실험할 모델 변경
EXPERIMENT_CONFIG = {
    'student_id': '20221555',
    'model_name': 'MiniNetV3_size',  # 다른 모델로 변경 가능
    'n_trials': 50,                  # trial 수 조정
    'timeout': None,                 # 시간 제한 (초)
}

# Search space 조정
HYPERPARAMETER_SPACE = {
    'learning_rate': {
        'type': 'float',
        'low': 1e-4,      # 최소값 조정
        'high': 1e-2,     # 최대값 조정
        'log': True
    },
    # ... 다른 파라미터들
}
```

### Trial 수 변경
```python
# 빠른 테스트용
EXPERIMENT_CONFIG['n_trials'] = 10

# 본격적인 최적화용  
EXPERIMENT_CONFIG['n_trials'] = 100
```

---

## 📊 결과 확인

### 실시간 모니터링
```bash
# 로그 실시간 확인
tail -f logs/optuna_YYMMDD_HHMMSS.log

# 프로세스 상태 확인
ps -p $(cat logs/optuna_YYMMDD_HHMMSS.pid)
```

### 결과 파일들
```
📁 결과 파일 위치:
├── semantic_segmentation_optimization_YYMMDD_HHMMSS.db  # Optuna SQLite DB
├── optuna_history_*.csv                                 # 최적화 히스토리  
├── output/output_YYMMDD_HHMMSS_trialBEST/              # 최적 결과
└── vis/OPTUNA_OUTPUTS_*/                               # 시각화 결과
```

### 결과 분석 스크립트
```python
# Optuna 결과 로드 및 분석
import optuna
import pandas as pd

# Study 로드
study = optuna.load_study(
    study_name="your_study_name",
    storage="sqlite:///your_study.db"
)

# 최적 파라미터 확인
print("Best parameters:", study.best_params)
print("Best IoU:", study.best_value)

# 전체 히스토리 확인
df = study.trials_dataframe()
print(df.head())
```

---

## 🛠️ 문제 해결

### 자주 발생하는 문제들

#### 1. 모델 클래스를 찾을 수 없음
```bash
ModuleNotFoundError: No module named 'submission_20221555'
```
**해결책:**
```bash
# models 디렉토리 구조 확인
ls -la models/submission_20221555/

# __init__.py 파일 존재 확인
ls -la models/submission_20221555/__init__.py

# 모델 클래스명 확인 (submission_MiniNetV3_size)
grep -n "class submission_" models/submission_20221555/*.py
```

#### 2. CUDA 메모리 부족
```bash
RuntimeError: CUDA out of memory
```
**해결책:**
```python
# optuna_config.py에서 병렬 실행 수 조정
EXPERIMENT_CONFIG['n_jobs'] = 1  # 1로 설정 (기본값)

# 또는 실험 전 GPU 메모리 정리
import torch
torch.cuda.empty_cache()
```

#### 3. 실험이 멈춘 것 같을 때
```bash
# 프로세스 확인
ps aux | grep optuna

# 로그 확인
tail -100 logs/optuna_*.log

# GPU 사용률 확인
nvidia-smi
```

### 실험 중단 방법
```bash
# PID로 중단
kill $(cat logs/optuna_YYMMDD_HHMMSS.pid)

# 강제 종료
kill -9 $(cat logs/optuna_YYMMDD_HHMMSS.pid)

# 또는 프로세스명으로 
pkill -f optuna_experiment.py
```

---

## 🎯 성능 최적화 팁

### 1. Search Space 최적화
```python
# 이미 잘 알려진 범위로 제한
'learning_rate': {
    'low': 5e-4,   # 너무 작은 값 제외
    'high': 5e-3,  # 너무 큰 값 제외
}

# 불필요한 옵션 제거
'scheduler': {
    'choices': ['warmup_cosine']  # 가장 효과적인 것만
}
```

### 2. Pruning 전략 조정
```python
# 더 공격적인 pruning
EARLY_STOPPING_CONFIG = {
    'voc_threshold': 0.02,  # VOC 임계값 높이기
    'patience_trials': 5,   # 빠른 중단
}
```

### 3. Trial 수 계획
```
🔥 단계별 접근:
1️⃣ 빠른 탐색: 20 trials (2-3시간)
2️⃣ 세밀한 조정: 50 trials (6-8시간)  
3️⃣ 최종 검증: 100 trials (12-15시간)
```

### 4. 병렬 실행 (고급)
```python
# 여러 GPU가 있는 경우
EXPERIMENT_CONFIG['n_jobs'] = 2  # GPU 수에 맞게 조정

# 주의: 각 job은 독립적인 GPU 메모리 필요
```

---

## 📈 성공 사례 및 벤치마크

### 예상 성능 개선
```
기존 수동 튜닝: 0.35 IoU (예시)
Optuna 최적화: 0.40+ IoU (목표)
최적화 시간: 8-12시간 (50 trials)
```

### 실제 실험 결과 (업데이트 예정)
```
Trial #15: IoU 0.3856 (best so far)
  - optimizer: adamw
  - learning_rate: 0.0067
  - scheduler: warmup_cosine
  - warmup_epochs: 7

Trial #32: IoU 0.4012 (new best!)
  - optimizer: adamw  
  - learning_rate: 0.0045
  - scheduler: warmup_cosine
  - warmup_epochs: 5
```

---

## 🔄 다음 단계

### 실험 완료 후 할 일
1. **결과 분석**: 최적 하이퍼파라미터 패턴 파악
2. **모델 업데이트**: training_args.py에 최적 설정 반영
3. **추가 실험**: 다른 모델에 동일한 최적화 적용
4. **문서화**: 발견한 인사이트 기록

### 확장 가능성
- **다중 모델 최적화**: 여러 모델 아키텍처 동시 비교
- **데이터셋별 특화**: 각 데이터셋에 최적화된 하이퍼파라미터 탐색
- **앙상블 최적화**: 여러 모델 조합 최적화

---

## 📞 지원 및 문의

### 로그 분석 요청 시 포함할 정보
```
1. 실행 명령어
2. 로그 파일 경로
3. 에러 메시지 (있는 경우)
4. 시스템 환경 (GPU, 메모리 등)
```

### 성능 개선 제안
- 새로운 하이퍼파라미터 추가
- Search space 개선 아이디어
- Pruning 전략 최적화

---

**🎉 성공적인 하이퍼파라미터 최적화를 위해 화이팅!** 