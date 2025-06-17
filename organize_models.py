#!/usr/bin/env python3
"""
모델 파일 정리 스크립트

성능과 중요도에 따라 모델들을 분류하여 legacy 폴더로 이동합니다.
"""

import os
import shutil
from pathlib import Path


def organize_models(models_dir="models/submission_20221555", dry_run=True):
    """
    모델 파일들을 정리합니다.
    
    Args:
        models_dir: 모델 디렉토리 경로
        dry_run: True면 실제 이동하지 않고 출력만 함
    """
    
    # 보존할 모델들 (현재 폴더에 유지)
    keep_models = {
        # 최고 성능 모델들
        "submission_MiniNetV3.py",          # 전체 최고 성능 (0.4500 IoU)
        "submission_test.py",               # 10k 제한 내 최고 (0.4333 IoU)  
        "submission_HWNetUltra_v5.py",      # 10k 제한 내 2위 (0.4051 IoU)
        "submission_MiniNetv2.py",          # MiniNet 중간 최고 (0.4729 IoU)
        
        # 현재 개발 중
        "submission_MegaNetV1.py",          # 최신 개발 모델 (0.53 목표)
        
        # 시리즈별 최신/최고 성능
        "submission_MicroNetv13.py",        # MicroNet 시리즈 최종 (0.3783 IoU)
        "submission_MicroNetv13_Ultimate.py", # Ultimate 버전
        
        # 혁신적 기술이 포함된 모델들
        "submission_HWNetUltra_v4.py",      # Ultra 시리즈 주요 버전
        "submission_HWNetUltra_v3.py",
        "submission_HWNetUltra_v2.py",
        "submission_HWNetUltra.py",
    }
    
    # Legacy로 이동할 모델들의 패턴
    legacy_patterns = [
        # 초기 실험 모델들
        "submission_Baseline.py",
        "submission_LCNet*.py",
        "submission_SINet.py", 
        "submission_LEDNet.py",
        "submission_ESPNet.py",
        "submission_UNet_basic.py",
        "submission_DeepLabV3plus.py",
        "submission_MicroUNet*.py",
        
        # 구버전 시리즈들
        "submission_MicroNetv1.py",
        "submission_MicroNetv2.py", 
        "submission_MicroNetv3.py",
        "submission_MicroNetv4.py",
        "submission_MicroNetv5.py",
        "submission_MicroNetv6.py",
        "submission_MicroNetv7*.py",
        "submission_MicroNetv8.py",
        "submission_MicroNetv9.py",
        "submission_MicroNetv10*.py",
        "submission_MicroNetv11*.py",
        "submission_MicroNetv12*.py",
        
        # 구버전 HWNet 시리즈들  
        "submission_HWNetv*.py",
        "submission_HWNetNano*.py",
        "submission_HWNetPico*.py", 
        "submission_HWNetFemto*.py",
        "submission_HWNetPlain*.py",
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ 모델 디렉토리가 존재하지 않습니다: {models_dir}")
        return
    
    # Legacy 폴더 생성
    legacy_dir = os.path.join(models_dir, "legacy")
    if not dry_run:
        os.makedirs(legacy_dir, exist_ok=True)
        print(f"📁 Legacy 폴더 생성: {legacy_dir}")
    
    # 현재 모델 파일들 확인
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.py')]
    
    print(f"🔍 총 {len(model_files)}개의 모델 파일 발견")
    print("\n" + "=" * 60)
    
    # 보존할 모델들
    print("🏆 **보존할 모델들 (현재 폴더 유지)**")
    kept_count = 0
    for model_file in sorted(model_files):
        if model_file in keep_models:
            print(f"   ✅ {model_file}")
            kept_count += 1
    
    print(f"\n📊 보존 예정: {kept_count}개")
    
    # Legacy로 이동할 모델들 확인
    print("\n📦 **Legacy로 이동할 모델들**")
    to_move = []
    
    for model_file in sorted(model_files):
        if model_file in keep_models:
            continue
            
        if model_file == "__init__.py":
            continue
            
        # 패턴 매칭으로 Legacy 대상 확인
        should_move = False
        for pattern in legacy_patterns:
            if pattern.endswith('*'):
                # 와일드카드 패턴
                prefix = pattern[:-1]
                if model_file.startswith(prefix):
                    should_move = True
                    break
            else:
                # 정확한 매칭
                if model_file == pattern:
                    should_move = True
                    break
        
        if should_move:
            to_move.append(model_file)
            print(f"   📦 {model_file}")
    
    print(f"\n📊 이동 예정: {len(to_move)}개")
    
    # 분류되지 않은 모델들 확인
    unclassified = []
    for model_file in sorted(model_files):
        if (model_file not in keep_models and 
            model_file not in to_move and 
            model_file != "__init__.py"):
            unclassified.append(model_file)
    
    if unclassified:
        print("\n❓ **분류되지 않은 모델들 (수동 확인 필요)**")
        for model_file in unclassified:
            print(f"   ❓ {model_file}")
            is_move = input("해당 모델 파일 이동을 진행하시겠습니까? (y/N): ")
            if is_move.lower() in ['y', 'yes']:
                to_move.append(model_file)
                print(f"   ✅ {model_file}")
    
    # 실제 이동 작업
    if to_move:
        print(f"\n{'='*60}")
        if dry_run:
            print("🔄 [DRY RUN] 실제 이동하지 않음")
        else:
            print("📦 Legacy 폴더로 이동 중...")
            
        moved_count = 0
        for model_file in to_move:
            src_path = os.path.join(models_dir, model_file)
            dst_path = os.path.join(legacy_dir, model_file)
            
            if dry_run:
                print(f"   🔄 [DRY RUN] {model_file} → legacy/")
            else:
                try:
                    shutil.move(src_path, dst_path)
                    print(f"   ✅ {model_file} → legacy/")
                    moved_count += 1
                except Exception as e:
                    print(f"   ❌ 이동 실패 {model_file}: {e}")
        
        if not dry_run:
            print(f"\n✅ {moved_count}개 모델을 legacy 폴더로 이동 완료")
    
    # 요약
    print(f"\n📋 **정리 요약**")
    print(f"   🏆 보존된 모델: {kept_count}개")
    print(f"   📦 이동된 모델: {len(to_move)}개")
    if unclassified:
        print(f"   ❓ 미분류 모델: {len(unclassified)}개")
    
    print(f"\n💾 정리 후 main 폴더 모델 수: {kept_count + len(unclassified)}개")


def main():
    """메인 함수"""
    print("🗂️ 모델 파일 정리 스크립트")
    print("=" * 60)
    
    # 먼저 dry run으로 확인
    print("1️⃣ 정리 계획 확인 (Dry Run)")
    organize_models(dry_run=True)
    
    print("\n" + "=" * 60)
    response = input("모델 파일 이동을 진행하시겠습니까? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n2️⃣ 실제 이동 진행")
        organize_models(dry_run=False)
        print("\n✅ 모델 정리 완료!")
    else:
        print("\n🚫 이동이 취소되었습니다.")


if __name__ == "__main__":
    main() 