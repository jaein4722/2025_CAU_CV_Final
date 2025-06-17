#!/usr/bin/env python3
"""
results.csv 기반 실험 기록 정리 스크립트

results.csv에 기록되지 않은 output과 vis 폴더들을 찾고 삭제합니다.
"""

import os
import re
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path


def parse_timestamp_from_dirname(dirname):
    """디렉토리명에서 타임스탬프를 추출합니다."""
    # output_YYMMDD_HHMMSS 또는 TEST_OUTPUTS_YYMMDD_HHMMSS 패턴에서 시간 추출
    pattern = r'(\d{6}_\d{6})$'
    match = re.search(pattern, dirname)
    if match:
        timestamp_str = match.group(1)
        return timestamp_str
    return None


def load_recorded_experiments(csv_file="results.csv"):
    """results.csv에서 기록된 실험들의 타임스탬프를 로드합니다."""
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} 파일이 존재하지 않습니다.")
        return set()
    
    try:
        df = pd.read_csv(csv_file)
        if 'Experiment_Time' not in df.columns:
            print(f"❌ CSV 파일에 'Experiment_Time' 컬럼이 없습니다.")
            return set()
        
        # 기록된 실험 타임스탬프들을 집합으로 변환
        recorded_timestamps = set(df['Experiment_Time'].astype(str))
        print(f"📊 CSV에서 {len(recorded_timestamps)}개의 실험 기록 발견")
        return recorded_timestamps
        
    except Exception as e:
        print(f"❌ CSV 파일 읽기 오류: {e}")
        return set()


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
            
        # 타임스탬프 문자열 직접 비교
        if vis_timestamp == output_timestamp:
            return vis_dirname
            
        # 시간 차이가 tolerance_seconds 이내인지 확인
        try:
            output_dt = datetime.strptime(f"20{output_timestamp}", "%Y%m%d_%H%M%S")
            vis_dt = datetime.strptime(f"20{vis_timestamp}", "%Y%m%d_%H%M%S")
            time_diff = abs((output_dt - vis_dt).total_seconds())
            if time_diff <= tolerance_seconds:
                return vis_dirname
        except:
            continue
    
    return None


def cleanup_unrecorded_experiments(output_dir="output", vis_dir="vis", csv_file="results.csv", dry_run=True):
    """
    results.csv에 기록되지 않은 실험 디렉토리들을 정리합니다.
    
    Args:
        output_dir: output 디렉토리 경로
        vis_dir: vis 디렉토리 경로
        csv_file: results.csv 파일 경로
        dry_run: True면 실제 삭제하지 않고 출력만 함
    """
    if not os.path.exists(output_dir):
        print(f"❌ output 디렉토리가 존재하지 않습니다: {output_dir}")
        return
        
    if not os.path.exists(vis_dir):
        print(f"❌ vis 디렉토리가 존재하지 않습니다: {vis_dir}")
        return
    
    # CSV에서 기록된 실험들 로드
    recorded_timestamps = load_recorded_experiments(csv_file)
    if not recorded_timestamps:
        print("⚠️  기록된 실험이 없습니다. 작업을 중단합니다.")
        return
    
    unrecorded_dirs = []
    
    print(f"🔍 {output_dir} 디렉토리에서 미기록 실험 검사 중...")
    
    # output 디렉토리 내의 모든 디렉토리 확인
    for output_dirname in os.listdir(output_dir):
        if not output_dirname.startswith('output_'):
            continue
            
        output_path = os.path.join(output_dir, output_dirname)
        if not os.path.isdir(output_path):
            continue
            
        # 타임스탬프 추출
        output_timestamp = parse_timestamp_from_dirname(output_dirname)
        if output_timestamp is None:
            print(f"   ❌ 타임스탬프 추출 실패: {output_dirname}")
            continue
        
        # CSV에 기록되어 있는지 확인
        if output_timestamp not in recorded_timestamps:
            print(f"⚠️  미기록 실험 발견: {output_dirname} (CSV에 없음)")
            
            # 매칭되는 vis 디렉토리 찾기
            matching_vis_dir = find_matching_vis_directory(output_timestamp, vis_dir)
            
            if matching_vis_dir:
                print(f"   🎯 매칭된 vis 디렉토리: {matching_vis_dir}")
                unrecorded_dirs.append({
                    'output_dir': output_path,
                    'output_name': output_dirname,
                    'vis_dir': os.path.join(vis_dir, matching_vis_dir),
                    'vis_name': matching_vis_dir,
                    'timestamp': output_timestamp
                })
            else:
                print(f"   ⚠️  매칭되는 vis 디렉토리를 찾을 수 없음")
                unrecorded_dirs.append({
                    'output_dir': output_path,
                    'output_name': output_dirname,
                    'vis_dir': None,
                    'vis_name': None,
                    'timestamp': output_timestamp
                })
        else:
            print(f"✅ 기록된 실험: {output_dirname}")
    
    if not unrecorded_dirs:
        print("✅ 미기록 실험 디렉토리가 없습니다!")
        return
    
    print(f"\n📋 총 {len(unrecorded_dirs)}개의 미기록 실험 디렉토리 발견")
    print(f"💾 CSV에 기록된 실험: {len(recorded_timestamps)}개")
    
    # 삭제 실행 또는 dry run
    total_size = 0
    for item in unrecorded_dirs:
        print(f"\n🗂️  처리할 디렉토리:")
        print(f"   📁 Output: {item['output_name']}")
        print(f"   📅 Timestamp: {item['timestamp']}")
        if item['vis_name']:
            print(f"   📊 Vis: {item['vis_name']}")
        else:
            print(f"   📊 Vis: (매칭된 디렉토리 없음)")
        
        # 디렉토리 크기 계산
        try:
            if os.path.exists(item['output_dir']):
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(item['output_dir'])
                          for filename in filenames)
                total_size += size
                print(f"   💾 Size: {size / (1024*1024):.1f} MB")
        except:
            pass
        
        if dry_run:
            print(f"   🔄 [DRY RUN] 삭제 예정")
        else:
            try:
                # output 디렉토리 삭제
                if os.path.exists(item['output_dir']):
                    shutil.rmtree(item['output_dir'])
                    print(f"   ✅ Output 디렉토리 삭제 완료: {item['output_name']}")
                
                # vis 디렉토리 삭제 (있는 경우)
                if item['vis_dir'] and os.path.exists(item['vis_dir']):
                    shutil.rmtree(item['vis_dir'])
                    print(f"   ✅ Vis 디렉토리 삭제 완료: {item['vis_name']}")
                    
            except Exception as e:
                print(f"   ❌ 삭제 중 오류 발생: {e}")
    
    print(f"\n📊 총 삭제 예정 용량: {total_size / (1024*1024):.1f} MB")
    
    if dry_run:
        print(f"\n💡 실제로 삭제하려면 dry_run=False로 설정하세요.")


def main():
    """메인 함수"""
    print("🧹 CSV 기반 실험 기록 정리 스크립트")
    print("=" * 50)
    
    # 먼저 dry run으로 확인
    print("1️⃣ 삭제 대상 확인 (Dry Run)")
    cleanup_unrecorded_experiments(dry_run=True)
    
    print("\n" + "=" * 50)
    response = input("실제로 삭제를 진행하시겠습니까? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n2️⃣ 실제 삭제 진행")
        cleanup_unrecorded_experiments(dry_run=False)
        print("\n✅ 정리 완료!")
    else:
        print("\n🚫 삭제가 취소되었습니다.")


if __name__ == "__main__":
    main() 