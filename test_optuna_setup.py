#!/usr/bin/env python3
"""
Optuna 실험 환경 테스트 스크립트
실제 실험 전에 설정 및 환경이 올바르게 구성되었는지 확인
"""

import sys
import os
import importlib
import torch
from datetime import datetime

def test_imports():
    """필수 라이브러리 import 테스트"""
    print("🔍 라이브러리 Import 테스트")
    
    try:
        import optuna
        print(f"  ✅ optuna {optuna.__version__}")
    except ImportError as e:
        print(f"  ❌ optuna 설치 필요: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ❌ pandas 설치 필요: {e}")
        return False
    
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
        print(f"  📱 CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  🎮 GPU 개수: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  ❌ torch 설치 필요: {e}")
        return False
    
    try:
        from competition_utils import Execute_Experiment
        print("  ✅ competition_utils")
    except ImportError as e:
        print(f"  ❌ competition_utils 문제: {e}")
        return False
    
    try:
        from training_args import Make_Optimizer, Make_LR_Scheduler, Make_Loss_Function
        print("  ✅ training_args")
    except ImportError as e:
        print(f"  ❌ training_args 문제: {e}")
        return False
    
    return True

def test_config():
    """설정 파일 테스트"""
    print("\n⚙️ 설정 파일 테스트")
    
    try:
        from optuna_config import (
            EXPERIMENT_CONFIG, HYPERPARAMETER_SPACE, 
            validate_config, DATASET_CONFIG
        )
        print("  ✅ optuna_config import 성공")
        
        # 설정 유효성 검사
        validate_config()
        print("  ✅ 설정 유효성 검사 통과")
        
        # 기본 설정 출력
        print(f"  📊 Target model: {EXPERIMENT_CONFIG['model_name']}")
        print(f"  🎯 Trial 수: {EXPERIMENT_CONFIG['n_trials']}")
        print(f"  🔍 Search space 크기: {len(HYPERPARAMETER_SPACE)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 설정 파일 오류: {e}")
        return False

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n🧠 모델 로딩 테스트")
    
    try:
        from optuna_config import EXPERIMENT_CONFIG
        student_id = EXPERIMENT_CONFIG['student_id']
        model_name = EXPERIMENT_CONFIG['model_name']
        
        model_name_full = f'submission_{model_name}'
        module_path = f"models.submission_{student_id}.{model_name_full}"
        
        print(f"  🔍 모듈 경로: {module_path}")
        
        module = importlib.import_module(module_path)
        model_class = getattr(module, model_name_full)
        
        print(f"  ✅ 모델 클래스 로드 성공: {model_class}")
        
        # 테스트 모델 생성
        test_model = model_class(3, 21)  # VOC용 (3 channels, 21 classes)
        
        # 파라미터 수 확인
        total_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
        print(f"  📏 총 파라미터 수: {total_params:,}")
        
        if total_params > 10000:
            print(f"  ⚠️ 경고: 파라미터 수가 제한(10,000)을 초과함!")
        else:
            print(f"  ✅ 파라미터 제한 준수")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 모델 로딩 실패: {e}")
        return False

def test_data_paths():
    """데이터셋 경로 테스트"""
    print("\n📁 데이터셋 경로 테스트")
    
    try:
        from optuna_config import DATASET_CONFIG
        dataset_root = DATASET_CONFIG['dataset_root']
        dataset_names = DATASET_CONFIG['dataset_names']
        
        print(f"  📂 데이터셋 루트: {dataset_root}")
        
        if not os.path.exists(dataset_root):
            print(f"  ❌ 데이터셋 루트 디렉토리가 존재하지 않음: {dataset_root}")
            return False
        
        missing_datasets = []
        for dataset_name in dataset_names:
            dataset_path = os.path.join(dataset_root, dataset_name)
            if os.path.exists(dataset_path):
                print(f"  ✅ {dataset_name}")
                
                # train/val/test 디렉토리 확인
                for split in ['train', 'val', 'test']:
                    split_path = os.path.join(dataset_path, split)
                    if os.path.exists(split_path):
                        orig_path = os.path.join(split_path, 'Originals')
                        mask_path = os.path.join(split_path, 'Masks')
                        
                        if os.path.exists(orig_path) and os.path.exists(mask_path):
                            orig_count = len(os.listdir(orig_path))
                            mask_count = len(os.listdir(mask_path))
                            print(f"      {split}: {orig_count} images, {mask_count} masks")
                        else:
                            print(f"      ⚠️ {split}: Originals 또는 Masks 폴더 없음")
            else:
                missing_datasets.append(dataset_name)
                print(f"  ❌ {dataset_name} - 경로 없음: {dataset_path}")
        
        if missing_datasets:
            print(f"  ⚠️ 누락된 데이터셋: {missing_datasets}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 데이터셋 경로 테스트 실패: {e}")
        return False

def test_optuna_functionality():
    """기본 Optuna 기능 테스트"""
    print("\n🔬 Optuna 기능 테스트")
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        
        # 간단한 study 생성 테스트
        def simple_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study(direction='minimize')
        study.optimize(simple_objective, n_trials=3)
        
        print(f"  ✅ Optuna 기본 기능 테스트 성공")
        print(f"  🎯 Best value: {study.best_value:.4f}")
        print(f"  📊 Best params: {study.best_params}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Optuna 기능 테스트 실패: {e}")
        return False

def test_directory_permissions():
    """디렉토리 권한 테스트"""
    print("\n📝 디렉토리 권한 테스트")
    
    try:
        # output 디렉토리 생성 테스트
        test_output_dir = "output/test_temp"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # 테스트 파일 작성
        test_file = os.path.join(test_output_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # 테스트 파일 읽기
        with open(test_file, 'r') as f:
            content = f.read()
        
        # 정리
        os.remove(test_file)
        os.rmdir(test_output_dir)
        
        print("  ✅ output 디렉토리 읽기/쓰기 권한 확인")
        
        # logs 디렉토리 테스트
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        print("  ✅ logs 디렉토리 생성 가능")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 디렉토리 권한 테스트 실패: {e}")
        return False

def main():
    """전체 테스트 실행"""
    print("🧪 Optuna 실험 환경 테스트 시작")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_model_loading,
        test_data_paths,
        test_optuna_functionality,
        test_directory_permissions
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  💥 {test_func.__name__} 예외 발생: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약")
    
    success_count = sum(results)
    total_count = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  {i+1}. {test_func.__name__}: {status}")
    
    print(f"\n🎯 전체 결과: {success_count}/{total_count} 통과")
    
    if success_count == total_count:
        print("🎉 모든 테스트 통과! Optuna 실험 준비 완료")
        print("\n📝 다음 단계:")
        print("  1. ./run_optuna.sh 실행")
        print("  2. 또는 python optuna_experiment.py 직접 실행")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 위의 오류를 해결 후 다시 시도하세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 