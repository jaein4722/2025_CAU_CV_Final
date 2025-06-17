#!/usr/bin/env python3
"""
Optuna 기반 하이퍼파라미터 최적화 실험 스크립트
competition_modified.ipynb의 구조를 그대로 모방하여 py 파일로 구현
"""

import os
import sys
import time
import importlib
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import natsort
import torch
import optuna
from optuna.integration import TensorBoardCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
import logging

# 기존 모듈 import
from competition_utils import *
from training_args import *

# Optuna 설정 import
from optuna_config import (
    EXPERIMENT_CONFIG, STUDY_CONFIG, HYPERPARAMETER_SPACE,
    EARLY_STOPPING_CONFIG, LOGGING_CONFIG, DATASET_CONFIG,
    PERFORMANCE_BENCHMARKS, validate_config
)

# Optuna 로깅 설정
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def get_dataset_threshold(dataset_name, dataset_index):
    """
    데이터셋별 IoU 임계치를 반환하는 함수
    
    Args:
        dataset_name: 데이터셋 이름 (VOC, ETIS, CVPPP, CFD, CarDD)
        dataset_index: 데이터셋 순서 (0~4)
    
    Returns:
        float: 해당 데이터셋의 IoU 임계치
    """
    base_thresholds = EARLY_STOPPING_CONFIG['dataset_thresholds']
    threshold_multiplier = EARLY_STOPPING_CONFIG['threshold_multiplier']
    
    # 기본 임계치 가져오기
    base_threshold = base_thresholds.get(dataset_name, 0.1)  # 기본값 0.1
    
    # Progressive thresholds 옵션 적용
    if EARLY_STOPPING_CONFIG['progressive_thresholds']:
        # 데이터셋 순서에 따라 임계치를 점진적으로 증가
        # 첫 번째 데이터셋: 기본값의 80%
        # 마지막 데이터셋: 기본값의 120%
        progress_factor = 0.8 + (0.4 * dataset_index / 4)  # 0.8 -> 1.2
        base_threshold *= progress_factor
    
    # 전역 multiplier 적용
    final_threshold = base_threshold * threshold_multiplier
    
    return final_threshold


def log_pruning_statistics(trial, dataset_name, current_iou, threshold, pruned=False):
    """Pruning 통계를 로깅하는 함수"""
    status = "PRUNED" if pruned else "PASSED"
    print(f"📊 Trial {trial.number} | {dataset_name} | IoU: {current_iou:.4f} | 임계치: {threshold:.4f} | 상태: {status}")


class OptunaTrainingArgs:
    """Optuna trial을 기반으로 하이퍼파라미터를 동적으로 생성하는 클래스"""
    
    def __init__(self, trial):
        self.trial = trial
    
    def make_optimizer(self, model):
        """trial을 기반으로 optimizer 생성 (config 파일 설정 사용)"""
        # config에서 optimizer choices 가져오기
        optimizer_config = HYPERPARAMETER_SPACE['optimizer']
        optimizer_name = self.trial.suggest_categorical('optimizer', optimizer_config['choices'])
        
        # config에서 learning rate 범위 가져오기
        lr_config = HYPERPARAMETER_SPACE['learning_rate']
        lr = self.trial.suggest_float('learning_rate', lr_config['low'], lr_config['high'], 
                                     log=lr_config.get('log', False))
        
        # config에서 weight decay 범위 가져오기
        wd_config = HYPERPARAMETER_SPACE['weight_decay']
        weight_decay = self.trial.suggest_float('weight_decay', wd_config['low'], wd_config['high'],
                                               log=wd_config.get('log', False))
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            # SGD인 경우에만 momentum 설정
            if 'momentum' in HYPERPARAMETER_SPACE:
                momentum_config = HYPERPARAMETER_SPACE['momentum']
                momentum = self.trial.suggest_float('momentum', momentum_config['low'], momentum_config['high'])
            else:
                momentum = 0.9  # 기본값
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def make_lr_scheduler(self, optimizer):
        """trial을 기반으로 learning rate scheduler 생성 (config 파일 설정 사용)"""
        scheduler_config = HYPERPARAMETER_SPACE['scheduler']
        scheduler_name = self.trial.suggest_categorical('scheduler', scheduler_config['choices'])
        
        if scheduler_name == "warmup_cosine":
            # config에서 warmup cosine 설정 가져오기
            warmup_epochs_config = HYPERPARAMETER_SPACE.get('warmup_epochs', {'low': 0, 'high': 10})
            warmup_epochs = self.trial.suggest_int('warmup_epochs', 
                                                  warmup_epochs_config['low'], warmup_epochs_config['high'])
            
            warmup_factor_config = HYPERPARAMETER_SPACE.get('warmup_factor', {'low': 0.05, 'high': 0.3})
            warmup_factor = self.trial.suggest_float('warmup_factor', 
                                                    warmup_factor_config['low'], warmup_factor_config['high'])
            
            eta_min_config = HYPERPARAMETER_SPACE.get('eta_min', {'low': 1e-7, 'high': 1e-5, 'log': True})
            eta_min = self.trial.suggest_float('eta_min', eta_min_config['low'], eta_min_config['high'],
                                              log=eta_min_config.get('log', False))
            
            lr_scheduler = WarmupCosineLR(
                optimizer,
                T_max=50,
                warmup_iters=warmup_epochs,
                warmup_factor=warmup_factor,
                eta_min=eta_min
            )
        elif scheduler_name == "warmup_poly":
            warmup_epochs_config = HYPERPARAMETER_SPACE.get('warmup_epochs_poly', {'low': 0, 'high': 8})
            warmup_epochs = self.trial.suggest_int('warmup_epochs_poly', 
                                                  warmup_epochs_config['low'], warmup_epochs_config['high'])
            
            power_config = HYPERPARAMETER_SPACE.get('poly_power', {'low': 0.7, 'high': 1.0})
            power = self.trial.suggest_float('poly_power', power_config['low'], power_config['high'])
            
            lr_scheduler = WarmupPolyLR(
                optimizer,
                T_max=DATASET_CONFIG['epochs'],
                warmup_epochs=warmup_epochs,
                power=power,
                eta_min=1e-6
            )
        elif scheduler_name == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return lr_scheduler
    
    def make_loss_function(self, number_of_classes):
        """trial을 기반으로 loss function 생성 (config 파일 설정 사용)"""
        BINARY_SEG = True if number_of_classes == 2 else False
        
        if BINARY_SEG and 'dice_weight' in HYPERPARAMETER_SPACE:
            dice_weight_config = HYPERPARAMETER_SPACE['dice_weight']
            dice_weight = self.trial.suggest_float('dice_weight', 
                                                  dice_weight_config['low'], dice_weight_config['high'])
            loss = DiceCELoss(weight=dice_weight, mode='binary')
        elif BINARY_SEG:
            # config에 dice_weight가 없으면 기본값 사용
            loss = DiceCELoss(weight=0.5, mode='binary')
        else:
            # multiclass의 경우 기존 설정 유지
            loss = UniformCBCE_LovaszProb(number_of_classes)
        
        return loss


def run_single_experiment(student_id=None, model_name=None, trial=None, save_results=False):
    """
    단일 실험 실행 함수 (competition_modified.ipynb의 메인 로직)
    
    Args:
        student_id: 학생 ID (None이면 config에서 가져옴)
        model_name: 모델 이름 (None이면 config에서 가져옴)
        trial: optuna trial 객체 (None이면 기본 설정 사용)
        save_results: 결과 저장 여부
    
    Returns:
        mean_iou: 5개 데이터셋의 평균 IoU
        results_df: 실험 결과 DataFrame
    """
    
    # 실험 시작 시간
    start_time = time.time()
    now = datetime.now()
    experiments_time = now.strftime("%y%m%d_%H%M%S")
    
    if trial is not None:
        experiments_time += f"_trial{trial.number}"
    
    print(f'=== Experiment Start Time: {experiments_time} ===')
    
    # 설정에서 기본값 가져오기
    if student_id is None:
        student_id = EXPERIMENT_CONFIG['student_id']
    if model_name is None:
        model_name = EXPERIMENT_CONFIG['model_name']
    
    # 기본 설정
    model_name_full = f'submission_{model_name}'
    module_path = f"models.submission_{student_id}.{model_name_full}"
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name_full)
    
    # 데이터셋 설정 (config에서 가져오기)
    Dataset_root = DATASET_CONFIG['dataset_root']
    Dataset_Name_list = DATASET_CONFIG['dataset_names']
    number_of_classes_dict = DATASET_CONFIG['num_classes']
    
    # 실험 설정 (config에서 가져오기)
    epochs = DATASET_CONFIG['epochs']
    EARLY_STOP = DATASET_CONFIG['early_stop_patience']
    batch_size = DATASET_CONFIG['batch_size']
    EXCLUDE_BACKGROUND = DATASET_CONFIG['exclude_background']
    THRESHOLD = DATASET_CONFIG['threshold']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 결과 저장 설정
    output_root = 'output'
    output_root = f'{output_root}/output_{experiments_time}'
    os.makedirs(output_root, exist_ok=True)
    
    # vis_root 설정 수정: save_results가 False여도 임시 디렉토리 생성
    if save_results:
        vis_root = f'vis/OPTUNA_OUTPUTS_{experiments_time}'
        os.makedirs(vis_root, exist_ok=True)
    else:
        # save_results가 False여도 임시 디렉토리 생성 (test 함수 에러 방지)
        vis_root = f'temp_vis_{experiments_time}'
        os.makedirs(vis_root, exist_ok=True)
    
    # 결과 DataFrame 초기화
    eval_columns = ['Experiment Time', 'Train Time', 'Dataset Name', 'Model Name', 
                   'Val Loss', 'Test Loss', 'IoU', 'Dice', 'Precision', 'Recall', 
                   'Total Params', 'Train-Prediction Time']
    df = pd.DataFrame(index=None, columns=eval_columns)
    
    # Optuna training args 초기화 (trial이 있는 경우)
    optuna_args = OptunaTrainingArgs(trial) if trial is not None else None
    
    iou_scores = []
    seed = 1
    
    # 각 데이터셋에 대해 실험 수행
    for j, Dataset_Name in enumerate(Dataset_Name_list):
        print(f'\n=== Dataset: {Dataset_Name} ({j+1}/{len(Dataset_Name_list)}) ===')
        control_random_seed(seed)
        
        # 데이터셋 경로 설정
        Dataset_dir = Dataset_root + '/' + Dataset_Name
        Height, Width = (256, 256)
        in_channels = 3
        number_of_classes = number_of_classes_dict[Dataset_Name]
        BINARY_SEG = True if number_of_classes == 2 else False
        exclude_background = EXCLUDE_BACKGROUND
        out_channels = 1 if BINARY_SEG else number_of_classes
        
        # 데이터 로드
        train_image_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/train/Originals/*"))
        train_target_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/train/Masks/*"))
        validation_image_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/val/Originals/*"))
        validation_target_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/val/Masks/*"))
        test_image_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/test/Originals/*"))
        test_target_path_list = natsort.natsorted(glob.glob(f"{Dataset_dir}/test/Masks/*"))
        
        print(f'train/val/test: {len(train_image_path_list)}/{len(validation_image_path_list)}/{len(test_image_path_list)}')
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = ImagesDataset(train_image_path_list, train_target_path_list)
        validation_dataset = ImagesDataset(validation_image_path_list, validation_target_path_list)
        test_dataset = ImagesDataset(test_image_path_list, test_target_path_list)
        
        train_loader = SegDataLoader(train_dataset, batch_size=batch_size, num_workers=4, 
                                   pin_memory=True, shuffle=True, drop_last=True, fill_last_batch=False)
        validation_loader = SegDataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        test_loader = SegDataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        # 모델 생성
        print(f'{model_name_full} Dataset: {Dataset_Name}) ({j+1}/{len(Dataset_Name_list)})')
        output_dir = output_root + f'/{model_name_full}_{Dataset_Name}'
        control_random_seed(seed)
        
        model = model_class(in_channels, out_channels)
        model = model.to(device)
        
        # 하이퍼파라미터 설정 (Optuna trial 또는 기본값)
        if optuna_args is not None:
            optimizer = optuna_args.make_optimizer(model)
            lr_scheduler = optuna_args.make_lr_scheduler(optimizer)
            criterion = optuna_args.make_loss_function(number_of_classes)
        else:
            # 기본 설정 사용
            optimizer = Make_Optimizer(model)
            lr_scheduler = Make_LR_Scheduler(optimizer)
            criterion = Make_Loss_Function(number_of_classes)
        
        # 시각화 디렉토리 설정
        vis_dataset_root = f"{vis_root}/{Dataset_Name}"
        os.makedirs(vis_dataset_root, exist_ok=True)
        
        # 실험 실행
        df = Execute_Experiment(
            model_name_full, model, Dataset_Name, train_loader, validation_loader, test_loader,
            optimizer, lr_scheduler, criterion, number_of_classes, df, epochs, device, output_dir,
            BINARY_SEG, exclude_background, out_channels, seed, THRESHOLD, EARLY_STOP, 
            save_results, vis_dataset_root, experiments_time
        )
        
        # save_results가 False인 경우 임시 시각화 파일들 정리
        if not save_results:
            import shutil
            if os.path.exists(vis_dataset_root):
                shutil.rmtree(vis_dataset_root)
        
        # IoU 점수 추출
        current_iou = df.iloc[-1]['IoU']  # 마지막으로 추가된 행의 IoU
        iou_scores.append(current_iou)
        
        print(f'Dataset {Dataset_Name} IoU: {current_iou:.4f}')
        
        # 🎯 각 데이터셋 완료 후 중간 평가 및 pruning
        if trial is not None:
            # 현재까지의 평균 IoU로 중간 보고
            current_mean_iou = np.mean(iou_scores)
            trial.report(current_mean_iou, j)
            
            # 데이터셋별 중간 pruning 체크 (config 설정에 따라)
            if EARLY_STOPPING_CONFIG['intermediate_pruning']:
                dataset_threshold = get_dataset_threshold(Dataset_Name, j)
                
                print(f'📊 Dataset {Dataset_Name} 평가: IoU {current_iou:.4f} (임계치: {dataset_threshold:.4f})')
                
                # 현재 데이터셋의 IoU가 임계치 미달인지 체크
                if current_iou < dataset_threshold:
                    print(f'⚠️  Dataset {Dataset_Name} IoU {current_iou:.4f} < 임계치 {dataset_threshold:.4f}')
                    
                    # Optuna pruner의 판단도 고려
                    if trial.should_prune():
                        print(f'🚫 Trial {trial.number} pruned after {Dataset_Name} (IoU: {current_iou:.4f} < {dataset_threshold:.4f})')
                        if not save_results and vis_root.startswith('temp_vis_'):
                            import shutil
                            if os.path.exists(vis_root):
                                shutil.rmtree(vis_root)
                        raise optuna.TrialPruned()
                    else:
                        print(f'🔄 임계치 미달이지만 pruner 판단으로 계속 진행 (Trial {trial.number})')
                else:
                    print(f'✅ Dataset {Dataset_Name} 임계치 통과: {current_iou:.4f} >= {dataset_threshold:.4f}')
            
            # 진행률 표시
            progress = ((j + 1) / len(Dataset_Name_list)) * 100
            print(f'📈 진행률: {progress:.1f}% ({j+1}/{len(Dataset_Name_list)} 데이터셋 완료)')
            print(f'🔢 현재까지 평균 IoU: {current_mean_iou:.4f}')
            print('-' * 60)
    
    # 평균 IoU 계산
    mean_iou = np.mean(iou_scores)
    print(f'\n=== Final Mean IoU: {mean_iou:.4f} ===')
    
    # 🎯 5개 데이터셋 완료 후 최종 pruning 체크 (선택적)
    if (trial is not None and EARLY_STOPPING_CONFIG['enabled'] and 
        mean_iou < EARLY_STOPPING_CONFIG.get('final_threshold', 0.05)):
        print(f"Trial {trial.number} completed but final Mean IoU too low: {mean_iou:.4f}")
        # 여기서는 TrialPruned를 발생시키지 않고, 단순히 낮은 점수 반환
    
    # 결과 저장
    if save_results:
        df.to_csv(output_root + '/' + f'Competition_{experiments_time}.csv', 
                 index=False, header=True, encoding="cp949")
    
    # save_results가 False인 경우 임시 vis_root 디렉토리 정리
    if not save_results and vis_root.startswith('temp_vis_'):
        import shutil
        if os.path.exists(vis_root):
            shutil.rmtree(vis_root)
    
    total_time = time.time() - start_time
    print(f'Total experiment time: {total_time/60:.2f} minutes')
    
    return mean_iou, df


def objective(trial):
    """Optuna objective 함수 (config 설정 사용)"""
    try:
        # config에서 설정 가져오기
        mean_iou, _ = run_single_experiment(trial=trial, save_results=LOGGING_CONFIG['save_intermediate_results'])
        return mean_iou
        
    except optuna.TrialPruned:
        # Pruning은 정상적인 최적화 과정이므로 re-raise
        print(f"🔥 Trial {trial.number} was pruned (정상적인 조기 종료)")
        raise
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # 실패한 trial에 대해서는 매우 낮은 점수 반환
        return 0.0


def main():
    """메인 실행 함수 (config 설정 사용)"""
    print("=== Optuna 하이퍼파라미터 최적화 실험 시작 ===")
    
    # 설정 파일 유효성 검사
    validate_config()
    print("✅ 설정 파일 유효성 검사 통과")
    
    # config에서 설정 가져오기
    print(f"📊 실험 설정:")
    print(f"  - 모델: {EXPERIMENT_CONFIG['model_name']}")
    print(f"  - Trial 수: {EXPERIMENT_CONFIG['n_trials']}")
    print(f"  - Timeout: {EXPERIMENT_CONFIG['timeout']}")
    print(f"  - Search space 크기: {len(HYPERPARAMETER_SPACE)}")
    
    # 데이터셋별 임계치 정보 미리 출력
    if EARLY_STOPPING_CONFIG['intermediate_pruning']:
        print(f"\n🎯 데이터셋별 pruning 임계치:")
        for i, dataset_name in enumerate(DATASET_CONFIG['dataset_names']):
            threshold = get_dataset_threshold(dataset_name, i)
            print(f"  {dataset_name}: {threshold:.4f}")
        print(f"  임계치 배수: {EARLY_STOPPING_CONFIG['threshold_multiplier']}")
        print(f"  Progressive 임계치: {EARLY_STOPPING_CONFIG['progressive_thresholds']}")
    else:
        print(f"\n⚠️  중간 pruning 비활성화됨 - 모든 데이터셋 완료 후 평가")
    
    # Optuna study 생성 (config 설정 사용)
    if STUDY_CONFIG['persistent_study']:
        # 지속적 스터디: 고정된 이름 사용
        study_name = STUDY_CONFIG['study_base_name']
        storage_name = f"sqlite:///{study_name}.db" if LOGGING_CONFIG['sqlite_storage'] else None
        print(f"🔄 지속적 스터디 모드: 기존 DB 재사용")
    else:
        # 일회성 스터디: 타임스탬프 이름 사용  
        study_name = f"semantic_segmentation_optimization_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        storage_name = f"sqlite:///{study_name}.db" if LOGGING_CONFIG['sqlite_storage'] else None
        print(f"🆕 새로운 스터디 모드: 새 DB 생성")
    
    # Sampler 생성
    sampler_name = STUDY_CONFIG['sampler']
    sampler_kwargs = STUDY_CONFIG['sampler_kwargs']
    
    if sampler_name == 'TPESampler':
        sampler = TPESampler(**sampler_kwargs)
    elif sampler_name == 'RandomSampler':
        sampler = RandomSampler(seed=sampler_kwargs.get('seed', 42))
    elif sampler_name == 'CmaEsSampler':
        sampler = CmaEsSampler(seed=sampler_kwargs.get('seed', 42))
    else:
        sampler = TPESampler(**sampler_kwargs)
    
    # Pruner 생성 (pruner별 올바른 파라미터 사용)
    pruner_name = STUDY_CONFIG['pruner']
    
    if pruner_name == 'MedianPruner':
        pruner_kwargs = STUDY_CONFIG['median_pruner_kwargs']
        pruner = MedianPruner(**pruner_kwargs)
    elif pruner_name == 'HyperbandPruner':
        pruner_kwargs = STUDY_CONFIG['hyperband_pruner_kwargs']
        pruner = HyperbandPruner(**pruner_kwargs)
    else:
        # 기본값: MedianPruner
        pruner_kwargs = STUDY_CONFIG['median_pruner_kwargs']
        pruner = MedianPruner(**pruner_kwargs)
    
    study = optuna.create_study(
        direction=STUDY_CONFIG['direction'],
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    
    print(f"Study name: {study_name}")
    if storage_name:
        print(f"Storage: {storage_name}")
    
    # 기존 스터디 정보 출력
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"📚 기존 스터디 발견: {existing_trials}개 trial 이미 완료")
        
        # 기존 최고 성능 출력
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"🏆 기존 최고 Mean IoU: {study.best_value:.4f} (Trial {study.best_trial.number})")
            print(f"🔧 기존 최고 파라미터:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        
        # 스터디 리셋 옵션
        if STUDY_CONFIG['reset_study_on_start']:
            print(f"⚠️  스터디 리셋 옵션 활성화 - 기존 데이터 삭제")
            import sqlite3
            if storage_name and storage_name.startswith('sqlite:///'):
                db_path = storage_name.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    os.remove(db_path)
                    print(f"🗑️  기존 DB 삭제: {db_path}")
                    # 새로운 스터디 생성
                    study = optuna.create_study(
                        direction=STUDY_CONFIG['direction'],
                        sampler=sampler,
                        pruner=pruner,
                        study_name=study_name,
                        storage=storage_name,
                        load_if_exists=False
                    )
    else:
        print(f"🆕 새로운 스터디 시작")
    
    # 최적화 실행 (config 설정 사용)
    n_trials = EXPERIMENT_CONFIG['n_trials']
    timeout = EXPERIMENT_CONFIG['timeout']
    total_planned_trials = existing_trials + n_trials
    print(f"🚀 최적화 시작: {n_trials}개 새 trial 실행 (총 {total_planned_trials}개 예상)")
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # 결과 출력
    print("\n=== 최적화 완료 ===")
    total_trials = len(study.trials)
    new_trials = total_trials - existing_trials
    print(f"Total trials in study: {total_trials} (새로 추가: {new_trials})")
    
    # Pruning 통계 계산
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"✅ Completed trials: {len(completed_trials)}")
    print(f"✂️  Pruned trials: {len(pruned_trials)}")
    print(f"❌ Failed trials: {len(failed_trials)}")
    
    if len(completed_trials) > 0:
        print(f"🏆 Best trial number: {study.best_trial.number}")
        print(f"🎯 Best value (Mean IoU): {study.best_value:.4f}")
        
        print("\n🔧 Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        # 개선 히스토리 (최근 몇 개 trial만)
        if STUDY_CONFIG['persistent_study'] and new_trials > 0:
            print(f"\n📈 이번 세션 개선 히스토리:")
            recent_completed = [t for t in completed_trials if t.number >= existing_trials]
            if recent_completed:
                for trial in recent_completed[-3:]:  # 최근 3개만
                    print(f"  Trial {trial.number}: {trial.value:.4f}")
            else:
                print(f"  이번 세션에서 완료된 trial 없음")
    else:
        print("⚠️  모든 trial이 완료되지 않았습니다.")
    
    # 데이터셋별 임계치 정보 출력
    if EARLY_STOPPING_CONFIG['intermediate_pruning']:
        print(f"\n📊 사용된 데이터셋별 임계치:")
        for i, dataset_name in enumerate(DATASET_CONFIG['dataset_names']):
            threshold = get_dataset_threshold(dataset_name, i)
            print(f"  {dataset_name}: {threshold:.4f}")
        print(f"  임계치 배수: {EARLY_STOPPING_CONFIG['threshold_multiplier']}")
    
    # 최적 하이퍼파라미터로 최종 실험 실행 (결과 저장)
    print("\n=== 최적 하이퍼파라미터로 최종 실험 실행 ===")
    
    # 최적 trial 객체 생성 (결과 저장용)
    class BestTrial:
        def __init__(self, best_params):
            self.params = best_params
            self.number = "BEST"
        
        def suggest_categorical(self, name, choices):
            return self.params[name]
        
        def suggest_float(self, name, low, high, log=False):
            return self.params[name]
        
        def suggest_int(self, name, low, high):
            return self.params[name]
    
    best_trial = BestTrial(study.best_params)
    
    # final_mean_iou, final_df = run_single_experiment(
    #     trial=best_trial, save_results=True
    # )
    
    print(f"\n=== 완료 ===")
    #print(f"Final Mean IoU with best parameters: {final_mean_iou:.4f}")
    
    # 최적화 히스토리 저장
    history_df = study.trials_dataframe()
    #history_df.to_csv(f'optuna_history_{study_name}.csv', index=False)
    #print(f"Optimization history saved to: optuna_history_{study_name}.csv")
    
    return study


if __name__ == "__main__":
    start_time = time.time()
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main() 
    
    import requests

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    webhook_url = "https://discord.com/api/webhooks/1382936206731640912/FnHMkJjZo5UrE-ZXevPA9SIP-a-GRswUHb4jb4gjIsD7_McJ_ZR-h-rHm920ON-hHfIn"

    payload = {
        "content": f'optuna 최적화 완료!\n소요 시간: {horus}시간 {minutes}분 {seconds}초'
    }

    requests.post(webhook_url, json=payload)