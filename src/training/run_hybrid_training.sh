#!/bin/bash

#=============================================================================
# Hybrid FC-NN + TabTransformer Training Script
# 3-Year Overall Survival Classification with Missing Modality Learning
#=============================================================================

# 경로 설정
DATA_DIR="../../data/processed"
RESULTS_DIR="../../results"
OUTPUT_DIR="${RESULTS_DIR}/hybrid_training_$(date +%Y%m%d_%H%M%S)"

# 데이터 파일
COX_TABLE="${DATA_DIR}/integrated_table_cox.parquet"
METH_TABLE="${DATA_DIR}/methylation_table.parquet"
CLINICAL="${DATA_DIR}/processed_clinical_data_for_methylation.parquet"  # 8,577명 (Union)
SPLITS="${DATA_DIR}/train_val_test_splits.json"

# 훈련 파라미터
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
DEVICE="cuda"

#=============================================================================
# 실행 전 확인
#=============================================================================

echo "=============================================================================="
echo "🚀 HYBRID FC-NN + TABTRANSFORMER TRAINING"
echo "=============================================================================="
echo "Started at: $(date)"
echo ""
echo "Configuration:"
echo "  Cox table:      ${COX_TABLE}"
echo "  Methylation:    ${METH_TABLE}"
echo "  Clinical data:  ${CLINICAL}"
echo "  Splits:         ${SPLITS}"
echo "  Output:         ${OUTPUT_DIR}"
echo ""
echo "Training parameters:"
echo "  Epochs:         ${EPOCHS}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Learning rate:  ${LEARNING_RATE}"
echo "  Device:         ${DEVICE}"
echo ""

# 데이터 파일 확인
echo "Checking required files..."
MISSING_FILES=0

if [ ! -f "${COX_TABLE}" ]; then
    echo "  ❌ Cox table not found: ${COX_TABLE}"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    COX_SIZE=$(du -h "${COX_TABLE}" | cut -f1)
    echo "  ✅ Cox table: ${COX_SIZE}"
fi

if [ ! -f "${METH_TABLE}" ]; then
    echo "  ❌ Methylation table not found: ${METH_TABLE}"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    METH_SIZE=$(du -h "${METH_TABLE}" | cut -f1)
    echo "  ✅ Methylation table: ${METH_SIZE}"
fi

if [ ! -f "${CLINICAL}" ]; then
    echo "  ❌ Clinical data not found: ${CLINICAL}"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    echo "  ✅ Clinical data found"
fi

if [ ! -f "${SPLITS}" ]; then
    echo "  ❌ Splits file not found: ${SPLITS}"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    echo "  ✅ Splits file found"
fi

if [ ${MISSING_FILES} -gt 0 ]; then
    echo ""
    echo "❌ Missing ${MISSING_FILES} required file(s)!"
    echo "   Please wait for integrated_dataset_builder.py to complete."
    echo "   Or run: ./src/preprocessing/run_integrated_dataset_builder.sh"
    exit 1
fi

echo ""
echo "✅ All required files found!"
echo ""

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  nvidia-smi not found. Using CPU mode."
    DEVICE="cpu"
fi

# Output 디렉토리 생성
mkdir -p "${OUTPUT_DIR}"

#=============================================================================
# 훈련 시작
#=============================================================================

echo "=============================================================================="
echo "🔥 Starting Training..."
echo "=============================================================================="
echo ""

# Python 스크립트 실행
python train_hybrid.py \
    --cox-table "${COX_TABLE}" \
    --meth-table "${METH_TABLE}" \
    --clinical "${CLINICAL}" \
    --splits "${SPLITS}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --device ${DEVICE}

EXIT_CODE=$?

#=============================================================================
# 결과 확인
#=============================================================================

echo ""
echo "=============================================================================="
echo "📊 Training Completed"
echo "=============================================================================="
echo "Finished at: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/"
    echo ""

    # Test results 출력
    if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
        echo "Test Results:"
        cat "${OUTPUT_DIR}/test_results.json"
        echo ""
    fi

    echo "Best model: ${OUTPUT_DIR}/best_model.pth"
    echo "Training history: ${OUTPUT_DIR}/training_history.json"
else
    echo "❌ Training failed with exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo "=============================================================================="
