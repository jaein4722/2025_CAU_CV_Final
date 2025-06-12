#!/usr/bin/env python3
"""
학습이 완료되지 않은 디렉토리 정리 스크립트

output 폴더에서 md 파일이 없는 디렉토리를 찾고,
해당 시간과 매칭되는 vis 폴더의 디렉토리를 함께 삭제합니다.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path


def parse_timestamp_from_dirname(dirname):
    """디렉토리명에서 타임스탬프를 추출합니다."""
    # output_YYMMDD_HHMMSS 또는 TEST_OUTPUTS_YYMMDD_HHMMSS 패턴에서 시간 추출
    pattern = r'(\d{6}_\d{6})$'
    match = re.search(pattern, dirname)
    if match:
        timestamp_str = match.group(1)
        # YYMMDD_HHMMSS 형태를 datetime으로 변환
        return datetime.strptime(f"20{timestamp_str}", "%Y%m%d_%H%M%S")
    return None


def has_md_file(directory_path):
    """디렉토리에 .md 파일이 있는지 확인합니다."""
    for file in os.listdir(directory_path):
        if file.endswith('.md'):
            return True
    return False


def find_matching_vis_directory(output_timestamp, vis_dir, tolerance_seconds=2):
    """output 시간과 매칭되는 vis 디렉토리를 찾습니다 (1-2초 오차 허용)."""
    for vis_dirname in os.listdir(vis_dir):
        if not vis_dirname.startswith('TEST_OUTPUTS_'):
            continue
            
        vis_path = os.path.join(vis_dir, vis_dirname)
        if not os.path.isdir(vis_path):
            continue
            
        vis_timestamp = parse_timestamp_from_dirname(vis_dirname)
        if vis_timestamp is None:
            continue
            
        # 시간 차이가 tolerance_seconds 이내인지 확인
        time_diff = abs((output_timestamp - vis_timestamp).total_seconds())
        if time_diff <= tolerance_seconds:
            return vis_dirname
    
    return None


def cleanup_incomplete_training_dirs(output_dir="output", vis_dir="vis", dry_run=True):
    """
    학습이 완료되지 않은 디렉토리들을 정리합니다.
    
    Args:
        output_dir: output 디렉토리 경로
        vis_dir: vis 디렉토리 경로  
        dry_run: True면 실제 삭제하지 않고 출력만 함
    """
    if not os.path.exists(output_dir):
        print(f"❌ output 디렉토리가 존재하지 않습니다: {output_dir}")
        return
        
    if not os.path.exists(vis_dir):
        print(f"❌ vis 디렉토리가 존재하지 않습니다: {vis_dir}")
        return
    
    incomplete_dirs = []
    
    print(f"🔍 {output_dir} 디렉토리에서 미완료 학습 검사 중...")
    
    # output 디렉토리 내의 모든 디렉토리 확인
    for output_dirname in os.listdir(output_dir):
        if not output_dirname.startswith('output_'):
            continue
            
        output_path = os.path.join(output_dir, output_dirname)
        if not os.path.isdir(output_path):
            continue
            
        # md 파일이 있는지 확인
        if not has_md_file(output_path):
            print(f"⚠️  미완료 학습 발견: {output_dirname} (md 파일 없음)")
            
            # 타임스탬프 추출
            output_timestamp = parse_timestamp_from_dirname(output_dirname)
            if output_timestamp is None:
                print(f"   ❌ 타임스탬프 추출 실패: {output_dirname}")
                continue
            
            # 매칭되는 vis 디렉토리 찾기
            matching_vis_dir = find_matching_vis_directory(output_timestamp, vis_dir)
            
            if matching_vis_dir:
                print(f"   🎯 매칭된 vis 디렉토리: {matching_vis_dir}")
                incomplete_dirs.append({
                    'output_dir': output_path,
                    'output_name': output_dirname,
                    'vis_dir': os.path.join(vis_dir, matching_vis_dir),
                    'vis_name': matching_vis_dir
                })
            else:
                print(f"   ⚠️  매칭되는 vis 디렉토리를 찾을 수 없음")
                incomplete_dirs.append({
                    'output_dir': output_path,
                    'output_name': output_dirname,
                    'vis_dir': None,
                    'vis_name': None
                })
    
    if not incomplete_dirs:
        print("✅ 미완료 학습 디렉토리가 없습니다!")
        return
    
    print(f"\n📋 총 {len(incomplete_dirs)}개의 미완료 학습 디렉토리 발견")
    
    # 삭제 실행 또는 dry run
    for item in incomplete_dirs:
        print(f"\n🗂️  처리할 디렉토리:")
        print(f"   📁 Output: {item['output_name']}")
        if item['vis_name']:
            print(f"   📊 Vis: {item['vis_name']}")
        else:
            print(f"   📊 Vis: (매칭된 디렉토리 없음)")
        
        if dry_run:
            print(f"   🔄 [DRY RUN] 삭제 예정")
        else:
            try:
                # output 디렉토리 삭제
                shutil.rmtree(item['output_dir'])
                print(f"   ✅ Output 디렉토리 삭제 완료: {item['output_name']}")
                
                # vis 디렉토리 삭제 (있는 경우)
                if item['vis_dir'] and os.path.exists(item['vis_dir']):
                    shutil.rmtree(item['vis_dir'])
                    print(f"   ✅ Vis 디렉토리 삭제 완료: {item['vis_name']}")
                    
            except Exception as e:
                print(f"   ❌ 삭제 중 오류 발생: {e}")
    
    if dry_run:
        print(f"\n💡 실제로 삭제하려면 dry_run=False로 설정하세요.")


def main():
    """메인 함수"""
    print("🧹 학습 미완료 디렉토리 정리 스크립트")
    print("=" * 50)
    
    # 먼저 dry run으로 확인
    print("1️⃣ 삭제 대상 확인 (Dry Run)")
    cleanup_incomplete_training_dirs(dry_run=True)
    
    print("\n" + "=" * 50)
    response = input("실제로 삭제를 진행하시겠습니까? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n2️⃣ 실제 삭제 진행")
        cleanup_incomplete_training_dirs(dry_run=False)
        print("\n✅ 정리 완료!")
    else:
        print("\n🚫 삭제가 취소되었습니다.")


if __name__ == "__main__":
    main() 