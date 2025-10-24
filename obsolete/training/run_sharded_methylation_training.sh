#!/bin/bash

# Sharded Methylation TabTransformer 훈련 스크립트
# Usage: ./run_sharded_methylation_training.sh [phase1|phase2|phase3|all]

set -e

# 기본 설정
TRAINING_DIR="/data4/workspace_vscode/CANCER_FOUNDATION_MODEL/src/training"
LOG_DIR="$TRAINING_DIR/../../logs"
PYTHON_SCRIPT="train_sharded_methylation.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 로그 함수
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# GPU 정보 확인
check_gpu() {
    log_info "GPU 상태 확인 중..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
        echo ""
    else
        log_warn "NVIDIA GPU를 찾을 수 없습니다."
    fi
}

# 메인 실행
main() {
    local phase=${1:-"phase1"}

    echo "========================================"
    echo "🔬 Sharded Methylation TabTransformer 훈련"
    echo "========================================"
    echo "시작 시간: $(date)"
    echo "Phase: $phase"
    echo "========================================"

    # GPU 상태 확인
    check_gpu

    # Python 환경 확인
    if ! command -v python &> /dev/null; then
        log_error "Python을 찾을 수 없습니다."
        exit 1
    fi

    # 훈련 스크립트 존재 확인
    if [ ! -f "$TRAINING_DIR/$PYTHON_SCRIPT" ]; then
        log_error "훈련 스크립트를 찾을 수 없습니다: $TRAINING_DIR/$PYTHON_SCRIPT"
        exit 1
    fi

    # 샤드 데이터 존재 확인
    SHARD_DIR="$TRAINING_DIR/../../data/processed/methylation_shards"
    if [ ! -d "$SHARD_DIR" ]; then
        log_error "샤드 데이터 디렉토리를 찾을 수 없습니다: $SHARD_DIR"
        log_info "먼저 데이터 샤딩을 실행하세요:"
        log_info "  cd src/preprocessing"
        log_info "  python shard_methylation_data.py"
        exit 1
    fi

    local log_file="$LOG_DIR/sharded_methylation_${phase}_${TIMESTAMP}.log"

    log_info "🧬 Sharded Methylation 훈련 시작 (Phase: $phase)"
    log_info "로그 파일: $log_file"

    # training 폴더로 이동하여 실행
    cd "$TRAINING_DIR"

    nohup python -u "$PYTHON_SCRIPT" \
        --phase "$phase" \
        --epochs_phase1 30 \
        --epochs_phase2 20 \
        --epochs_phase3 10 \
        --batch_size 16 \
        --lr 1e-4 \
        --seed 42 \
        --shard_dir "../../data/processed/methylation_shards" \
        --clinical_data "../../data/processed/processed_clinical_data_for_methylation.parquet" \
        --checkpoint_dir "./checkpoints/methylation" \
        --results_dir "../../results" \
        > "$log_file" 2>&1 &

    local pid=$!

    log_info "훈련 시작됨 (PID: $pid)"
    log_info "진행 상황 모니터링: tail -f $log_file"
    log_info "훈련 중지: kill $pid"

    # 백그라운드 프로세스 대기 (옵션)
    # wait $pid

    echo ""
    echo "========================================"
    echo "📊 훈련 정보"
    echo "========================================"
    echo "PID: $pid"
    echo "로그: $log_file"
    echo "체크포인트: ./checkpoints/methylation/seed_42/"
    echo "========================================"
}

# Ctrl+C 처리
cleanup() {
    echo ""
    log_warn "훈련 중지 신호 수신됨..."
    log_info "정리 완료. 스크립트 종료."
    exit 1
}

trap cleanup SIGINT SIGTERM

# 스크립트 실행
main "$@"
