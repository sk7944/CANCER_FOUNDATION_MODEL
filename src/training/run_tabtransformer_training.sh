#!/bin/bash

# TabTransformer 훈련 스크립트 (백그라운드 실행 with 로깅)
# Usage: ./run_tabtransformer_training.sh [cox|methylation|both] [ensemble|single]

set -e  # 오류 발생 시 스크립트 종료

# 기본 설정
TRAINING_DIR="/data4/workspace_vscode/CANCER_FOUNDATION_MODEL/src/training"
LOG_DIR="$TRAINING_DIR/../../logs"
PYTHON_SCRIPT="train_tabtransformer.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# CoxTabTransformer 훈련 함수
train_cox_model() {
    local ensemble_mode=${1:-"single"}
    local log_file="$LOG_DIR/cox_tabtransformer_${TIMESTAMP}.log"
    local pid_file="$LOG_DIR/cox_training.pid"
    
    log_info "🧬 CoxTabTransformer 훈련 시작"
    if [ "$ensemble_mode" = "ensemble" ]; then
        log_info "🎯 앙상블 모드 활성화 (5개 시드)"
    fi
    log_info "로그 파일: $log_file"
    
    # training 폴더로 이동하여 실행
    cd "$TRAINING_DIR"
    
    # 앙상블 모드 여부에 따라 다른 명령 실행
    if [ "$ensemble_mode" = "ensemble" ]; then
        nohup python "$PYTHON_SCRIPT" \
            --model cox \
            --ensemble \
            --n_seeds 5 \
            --epochs 50 \
            --batch_size 32 \
            --lr 2e-5 \
            --data_dir "../../data/processed" \
            --results_dir "../../results" \
            --checkpoint_dir "./checkpoints" \
            --target_auc 0.85 \
            > "$log_file" 2>&1 &
    else
        nohup python "$PYTHON_SCRIPT" \
            --model cox \
            --epochs 50 \
            --batch_size 32 \
            --lr 2e-5 \
            --data_dir "../../data/processed" \
            --results_dir "../../results" \
            --checkpoint_dir "./checkpoints" \
            > "$log_file" 2>&1 &
    fi
    
    local cox_pid=$!
    echo $cox_pid > "$pid_file"
    
    log_info "CoxTabTransformer 훈련 시작됨 (PID: $cox_pid)"
    log_info "진행 상황 모니터링: tail -f $log_file"
    log_info "훈련 중지: kill $cox_pid"
    
    return $cox_pid
}

# MethylationTabTransformer 훈련 함수  
train_methylation_model() {
    local ensemble_mode=${1:-"single"}
    local log_file="$LOG_DIR/methylation_tabtransformer_${TIMESTAMP}.log"
    local pid_file="$LOG_DIR/methylation_training.pid"
    
    log_info "🔬 MethylationTabTransformer 훈련 시작"
    if [ "$ensemble_mode" = "ensemble" ]; then
        log_info "🎯 앙상블 모드 활성화 (5개 시드)"
    fi
    log_info "로그 파일: $log_file"
    
    # training 폴더로 이동하여 실행
    cd "$TRAINING_DIR"
    
    # 앙상블 모드 여부에 따라 다른 명령 실행
    if [ "$ensemble_mode" = "ensemble" ]; then
        nohup python "$PYTHON_SCRIPT" \
            --model methylation \
            --ensemble \
            --n_seeds 5 \
            --epochs 30 \
            --batch_size 16 \
            --lr 2e-5 \
            --data_dir "../../data/processed" \
            --results_dir "../../results" \
            --checkpoint_dir "./checkpoints" \
            --target_auc 0.85 \
            > "$log_file" 2>&1 &
    else
        nohup python "$PYTHON_SCRIPT" \
            --model methylation \
            --epochs 30 \
            --batch_size 16 \
            --lr 2e-5 \
            --data_dir "../../data/processed" \
            --results_dir "../../results" \
            --checkpoint_dir "./checkpoints" \
            > "$log_file" 2>&1 &
    fi
    
    local meth_pid=$!
    echo $meth_pid > "$pid_file"
    
    log_info "MethylationTabTransformer 훈련 시작됨 (PID: $meth_pid)"
    log_info "진행 상황 모니터링: tail -f $log_file"
    log_info "훈련 중지: kill $meth_pid"
    
    return $meth_pid
}

# 훈련 상태 모니터링
monitor_training() {
    local pid=$1
    local model_name=$2
    
    while kill -0 $pid 2>/dev/null; do
        sleep 30
        log_info "$model_name 훈련 진행 중... (PID: $pid)"
    done
    
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "✅ $model_name 훈련 완료!"
    else
        log_error "❌ $model_name 훈련 실패 (Exit code: $exit_code)"
    fi
    
    return $exit_code
}

# 메인 실행 함수
main() {
    local mode=${1:-"both"}
    local ensemble_mode=${2:-"single"}
    
    echo "========================================"
    echo "🧬 TabTransformer 훈련 스크립트"
    echo "========================================"
    echo "시작 시간: $(date)"
    echo "모드: $mode"
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
    
    case $mode in
        "cox")
            train_cox_model "$ensemble_mode"
            cox_pid=$?
            monitor_training $cox_pid "CoxTabTransformer"
            ;;
        "methylation") 
            train_methylation_model "$ensemble_mode"
            meth_pid=$?
            monitor_training $meth_pid "MethylationTabTransformer"
            ;;
        "both")
            log_info "🔄 순차 훈련: Cox → Methylation"
            
            # Cox 모델 먼저 훈련
            train_cox_model "$ensemble_mode"
            cox_pid=$?
            log_info "Cox 모델 훈련 완료 대기 중..."
            monitor_training $cox_pid "CoxTabTransformer"
            cox_result=$?
            
            if [ $cox_result -eq 0 ]; then
                log_info "Cox 모델 훈련 성공! Methylation 모델 시작..."
                sleep 5
                
                # Methylation 모델 훈련
                train_methylation_model "$ensemble_mode"  
                meth_pid=$?
                monitor_training $meth_pid "MethylationTabTransformer"
                meth_result=$?
                
                if [ $meth_result -eq 0 ]; then
                    log_info "🎉 모든 모델 훈련 완료!"
                else
                    log_error "Methylation 모델 훈련 실패"
                    exit 1
                fi
            else
                log_error "Cox 모델 훈련 실패"
                exit 1
            fi
            ;;
        *)
            log_error "잘못된 모드: $mode (사용 가능: cox, methylation, both)"
            echo ""
            echo "사용법:"
            echo "  $0 cox          # Cox 모델만 훈련"
            echo "  $0 methylation  # Methylation 모델만 훈련"  
            echo "  $0 both         # 두 모델 순차 훈련 (기본값)"
            exit 1
            ;;
    esac
    
    echo ""
    echo "========================================"
    echo "📊 훈련 완료 정보"
    echo "========================================"
    echo "완료 시간: $(date)"
    echo "로그 디렉토리: $LOG_DIR"
    echo "결과 디렉토리: ../../results"
    echo "========================================"
}

# Ctrl+C 처리
cleanup() {
    echo ""
    log_warn "훈련 중지 신호 수신됨..."
    
    # 실행 중인 훈련 프로세스 종료
    if [ -f "$LOG_DIR/cox_training.pid" ]; then
        cox_pid=$(cat "$LOG_DIR/cox_training.pid")
        if kill -0 $cox_pid 2>/dev/null; then
            log_warn "Cox 훈련 프로세스 종료 중... (PID: $cox_pid)"
            kill $cox_pid
        fi
        rm -f "$LOG_DIR/cox_training.pid"
    fi
    
    if [ -f "$LOG_DIR/methylation_training.pid" ]; then
        meth_pid=$(cat "$LOG_DIR/methylation_training.pid")
        if kill -0 $meth_pid 2>/dev/null; then
            log_warn "Methylation 훈련 프로세스 종료 중... (PID: $meth_pid)"
            kill $meth_pid
        fi
        rm -f "$LOG_DIR/methylation_training.pid"
    fi
    
    log_info "정리 완료. 스크립트 종료."
    exit 1
}

trap cleanup SIGINT SIGTERM

# 스크립트 실행
main "$@"