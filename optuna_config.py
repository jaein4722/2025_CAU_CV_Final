"""
Optuna 실험 설정 파일
하이퍼파라미터 search space 및 실험 설정을 관리
"""

# 기본 실험 설정
EXPERIMENT_CONFIG = {
    'student_id': '20221555',
    'model_name': 'final',  # 실험할 모델명 (submission_ 접두사 제외)
    'n_trials': 100,                  # Optuna trial 수
    'timeout': None,                 # 실험 timeout (초, None이면 무제한)
    'n_jobs': 1,                     # 병렬 실행 수 (GPU 메모리 고려)
}

# Optuna Study 설정
STUDY_CONFIG = {
    'direction': 'maximize',         # IoU 최대화
    'sampler': 'TPESampler',         # TPE, RandomSampler, CmaEsSampler 중 선택
    'pruner': 'HyperbandPruner',        # MedianPruner, HyperbandPruner 중 선택
    'sampler_kwargs': {
        'seed': 1,
        'n_startup_trials': 5,
    },
    # Pruner별 설정 (pruner 타입에 따라 다른 파라미터 사용)
    'median_pruner_kwargs': {
        'n_startup_trials': 5,
        'n_warmup_steps': 2,
        'interval_steps': 1,
    },
    'hyperband_pruner_kwargs': {
        'min_resource': 1,        # 최소 epoch 수  
        'max_resource': 30,       # 최대 epoch 수
        'reduction_factor': 3,    # 감소 인자
    },
    
    # 지속적 스터디 설정 (같은 DB 재사용)
    'persistent_study': True,              # True: 고정 이름 사용, False: 타임스탬프 이름 사용
    'study_base_name': 'semantic_seg_opt', # 고정 스터디 이름 (persistent_study=True일 때)
    'reset_study_on_start': True,         # True시 기존 스터디 초기화 (주의!)
}

# 하이퍼파라미터 Search Space 정의
HYPERPARAMETER_SPACE = {
    # Optimizer 관련
    'optimizer': {
        'type': 'categorical',
        'choices': ['adam', 'adamw']
    },
    'learning_rate': {
        'type': 'float',
        'low': 1e-4,
        'high': 1e-2,
        'log': True
    },
    'weight_decay': {
        'type': 'float', 
        'low': 1e-5,
        'high': 1e-3,
        'log': True
    },
    
    # SGD 전용 (optimizer가 sgd일 때만 사용)
    'momentum': {
        'type': 'float',
        'low': 0.8,
        'high': 0.99,
        'condition': 'optimizer == sgd'
    },
    
    # Scheduler 관련
    'scheduler': {
        'type': 'categorical',
        'choices': ['warmup_cosine', 'warmup_poly', 'constant']
    },
    
    # WarmupCosineLR 전용
    'warmup_epochs': {
        'type': 'int',
        'low': 0,
        'high': 10,
        'condition': 'scheduler == warmup_cosine'
    },
    'warmup_factor': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'condition': 'scheduler == warmup_cosine'
    },
    'eta_min': {
        'type': 'float',
        'low': 1e-7,
        'high': 1e-5,
        'log': True,
        'condition': 'scheduler == warmup_cosine'
    },
    
    # WarmupPolyLR 전용  
    'warmup_epochs_poly': {
        'type': 'int',
        'low': 0,
        'high': 8,
        'condition': 'scheduler == warmup_poly'
    },
    'poly_power': {
        'type': 'float',
        'low': 0.7,
        'high': 1.0,
        'condition': 'scheduler == warmup_poly'
    },
    
    # Loss Function 관련 (Binary segmentation만)
    'dice_weight': {
        'type': 'float',
        'low': 0.3,
        'high': 0.7,
        'condition': 'binary_segmentation == True'
    }
}

# 조기 종료 설정
EARLY_STOPPING_CONFIG = {
    'enabled': True,
    'patience_trials': 10,           # 연속으로 개선되지 않는 trial 수
    'min_improvement': 0.001,        # 최소 개선폭 (IoU)
    'final_threshold': 0.05,         # 5개 데이터셋 완료 후 최종 Mean IoU 최소값
    
    # 데이터셋별 중간 pruning 설정
    'intermediate_pruning': True,    # 각 데이터셋별 중간 pruning 활성화
    'dataset_thresholds': {          # 각 데이터셋별 최소 IoU 임계치
        'VOC': 0.10,        # 다중 클래스(21개)로 가장 어려움 - 낮은 임계치
        'ETIS': 0.37,       # 의료 이미지, binary 하지만 까다로움
        'CVPPP': 0.90,      # 식물 이미지, binary, 중간 난이도
        'CFD': 0.37,        # 균열 탐지, binary, 상대적으로 쉬움
        'CarDD': 0.37,      # 차량 손상, binary, 중간 난이도
    },
    'progressive_thresholds': False, # 데이터셋 순서에 따라 임계치 점진적 증가 여부
    'threshold_multiplier': 1.0,    # 임계치 조정 배수 (실험적 조정용)
}

# 로깅 및 저장 설정
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'save_intermediate_results': False,  # 각 trial마다 결과 저장 여부
    'save_best_only': True,              # 최고 성능만 저장 여부
    'tensorboard_log': False,            # TensorBoard 로깅 여부
    'sqlite_storage': True,              # SQLite DB 저장 여부
}

# 데이터셋 설정 (고정값)
DATASET_CONFIG = {
    'dataset_root': 'Datasets',
    'dataset_names': ['VOC', 'ETIS', 'CVPPP', 'CFD', 'CarDD'],
    'num_classes': {'VOC': 21, 'ETIS': 2, 'CVPPP': 2, 'CFD': 2, 'CarDD': 2},
    'epochs': 30,
    'batch_size': 16,
    'early_stop_patience': 100,
    'threshold': 0.5,
    'exclude_background': True,
}

# 성능 벤치마크 (기존 실험 결과)
PERFORMANCE_BENCHMARKS = {
    'baseline': {
        'model': 'MiniNetV3',
        'mean_iou': 0.35,  # 예시값, 실제 값으로 업데이트 필요
        'params': 6000,
    },
    'current_best': {
        'model': 'test_model',
        'mean_iou': 0.4333,
        'params': 20274,
    },
    'target': {
        'mean_iou': 0.45,  # 목표 성능
        'params': 10000,   # 최대 파라미터 수
    }
}

def get_search_space_for_trial(trial):
    """
    Trial에 따라 동적으로 search space를 생성하는 함수
    조건부 하이퍼파라미터를 처리
    """
    params = {}
    
    for param_name, config in HYPERPARAMETER_SPACE.items():
        # 조건 체크
        if 'condition' in config:
            condition = config['condition']
            # 간단한 조건 파싱 (실제로는 더 복잡한 로직 필요할 수 있음)
            if 'optimizer == sgd' in condition and params.get('optimizer') != 'sgd':
                continue
            elif 'scheduler == warmup_cosine' in condition and params.get('scheduler') != 'warmup_cosine':
                continue
            elif 'scheduler == warmup_poly' in condition and params.get('scheduler') != 'warmup_poly':
                continue
            # binary_segmentation 조건은 데이터셋에 따라 다르므로 실행 시 처리
        
        # 파라미터 타입에 따라 suggest
        if config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, config['choices'])
        elif config['type'] == 'float':
            params[param_name] = trial.suggest_float(
                param_name, config['low'], config['high'], 
                log=config.get('log', False)
            )
        elif config['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, config['low'], config['high'])
    
    return params

def validate_config():
    """설정 파일의 유효성을 검사하는 함수"""
    errors = []
    
    # 필수 설정 체크
    if not EXPERIMENT_CONFIG.get('student_id'):
        errors.append("student_id가 설정되지 않았습니다.")
    
    if not EXPERIMENT_CONFIG.get('model_name'):
        errors.append("model_name이 설정되지 않았습니다.")
    
    if EXPERIMENT_CONFIG.get('n_trials', 0) <= 0:
        errors.append("n_trials는 양수여야 합니다.")
    
    # Search space 유효성 체크
    for param_name, config in HYPERPARAMETER_SPACE.items():
        if config['type'] == 'float' or config['type'] == 'int':
            if config['low'] >= config['high']:
                errors.append(f"{param_name}: low 값이 high 값보다 크거나 같습니다.")
    
    # 데이터셋별 임계치 유효성 체크
    if EARLY_STOPPING_CONFIG.get('intermediate_pruning', False):
        dataset_thresholds = EARLY_STOPPING_CONFIG.get('dataset_thresholds', {})
        dataset_names = DATASET_CONFIG['dataset_names']
        
        # 모든 데이터셋에 대한 임계치가 설정되었는지 확인
        for dataset_name in dataset_names:
            if dataset_name not in dataset_thresholds:
                errors.append(f"데이터셋 '{dataset_name}'에 대한 임계치가 설정되지 않았습니다.")
        
        # 임계치 값이 합리적 범위인지 확인
        for dataset_name, threshold in dataset_thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                errors.append(f"데이터셋 '{dataset_name}'의 임계치 {threshold}가 0.0~1.0 범위를 벗어났습니다.")
        
        # threshold_multiplier 유효성 체크
        multiplier = EARLY_STOPPING_CONFIG.get('threshold_multiplier', 1.0)
        if not (0.1 <= multiplier <= 5.0):
            errors.append(f"threshold_multiplier {multiplier}가 0.1~5.0 범위를 벗어났습니다.")
    
    if errors:
        raise ValueError("설정 오류:\n" + "\n".join(errors))
    
    return True

# 설정 유효성 검사 실행 
if __name__ == "__main__":
    try:
        validate_config()
        print("✅ 설정 파일 유효성 검사 통과")
        print(f"📊 실험 설정: {EXPERIMENT_CONFIG['n_trials']}개 trial")
        print(f"🎯 목표 모델: {EXPERIMENT_CONFIG['model_name']}")
        print(f"🔍 Search space 파라미터 수: {len(HYPERPARAMETER_SPACE)}")
    except ValueError as e:
        print(f"❌ 설정 오류: {e}") 