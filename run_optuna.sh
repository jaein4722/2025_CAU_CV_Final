#!/bin/bash

# Optuna 하이퍼파라미터 최적화 실험 실행 스크립트
# 백그라운드 실행 및 로그 기록 지원

# 스크립트 설정
SCRIPT_NAME="optuna_experiment.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/optuna_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/optuna_${TIMESTAMP}.pid"

# 로그 디렉토리 생성
mkdir -p ${LOG_DIR}

echo "=== Optuna 하이퍼파라미터 최적화 실험 시작 ==="
echo "시작 시간: $(date)"
echo "로그 파일: ${LOG_FILE}"
echo "PID 파일: ${PID_FILE}"

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU 정보 ==="
    nvidia-smi
    echo ""
fi

# Python 환경 확인
echo "=== Python 환경 정보 ==="
echo "Python 버전: $(python --version)"
echo "PyTorch 버전: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 사용 가능: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "CUDA 디바이스 수: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi
echo ""

# 실험 실행 (백그라운드)
echo "=== 실험 실행 시작 ==="
echo "백그라운드에서 실행 중..."

# nohup을 사용하여 백그라운드 실행
nohup python ${SCRIPT_NAME} > ${LOG_FILE} 2>&1 &

# PID 저장
echo $! > ${PID_FILE}
PID=$(cat ${PID_FILE})

echo "프로세스 ID: ${PID}"
echo "실험이 백그라운드에서 실행 중입니다."
echo ""

# 진행 상황 모니터링 함수
monitor_progress() {
    echo "=== 진행 상황 모니터링 ==="
    echo "실시간 로그 확인: tail -f ${LOG_FILE}"
    echo "프로세스 상태 확인: ps -p ${PID}"
    echo "실험 중단: kill ${PID}"
    echo ""
    
    # 처음 몇 줄의 로그 출력
    echo "=== 초기 로그 출력 (10초 후) ==="
    sleep 10
    if [ -f ${LOG_FILE} ]; then
        head -20 ${LOG_FILE}
        echo "..."
        echo "(더 많은 로그를 보려면: tail -f ${LOG_FILE})"
    else
        echo "로그 파일이 아직 생성되지 않았습니다."
    fi
}

# 모니터링 시작
monitor_progress

echo ""
echo "=== 실행 완료 정보 ==="
echo "실험 상태 확인: ps -p ${PID}"
echo "로그 전체 보기: cat ${LOG_FILE}"
echo "실험 중단: kill ${PID}"
echo "실험 강제 종료: kill -9 ${PID}" 