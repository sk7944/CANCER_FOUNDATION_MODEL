#!/bin/bash
# =============================================================================
# WSI Preprocessing Pipeline
# =============================================================================
#
# This script runs the WSI preprocessing pipeline to extract features from
# Whole Slide Images for survival prediction model training.
#
# Pipeline:
#   1. Tissue detection and segmentation
#   2. Patch extraction (256x256)
#   3. Stain normalization (Macenko method)
#   4. Feature extraction (Swin Transformer)
#   5. Save features to HDF5
#
# Usage:
#   ./run_preprocessing.sh [OPTIONS]
#
# Options:
#   --input DIR       Input directory containing SVS files (default: ../../data/raw)
#   --output DIR      Output directory for features (default: ../../data/processed)
#   --model NAME      Feature extraction model (default: swin_tiny)
#   --patch-size N    Patch size (default: 256)
#   --batch-size N    Batch size for feature extraction (default: 64)
#   --no-stain-norm   Disable stain normalization
#   --device DEVICE   CUDA device (default: cuda:0)
#
# =============================================================================

set -e

# Default parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

INPUT_DIR="${PROJECT_ROOT}/wsi_model/data/raw"
OUTPUT_DIR="${PROJECT_ROOT}/wsi_model/data/processed"
MODEL_NAME="swin_tiny"
PATCH_SIZE=256
BATCH_SIZE=64
STAIN_NORM="--stain-normalize"
DEVICE="cuda:0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --patch-size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-stain-norm)
            STAIN_NORM=""
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input DIR       Input directory containing SVS files"
            echo "  --output DIR      Output directory for features"
            echo "  --model NAME      Feature extraction model (swin_tiny, resnet50, etc.)"
            echo "  --patch-size N    Patch size (default: 256)"
            echo "  --batch-size N    Batch size for feature extraction"
            echo "  --no-stain-norm   Disable stain normalization"
            echo "  --device DEVICE   CUDA device (default: cuda:0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=============================================="
echo "WSI Preprocessing Pipeline"
echo "=============================================="
echo "Input directory:  ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model:            ${MODEL_NAME}"
echo "Patch size:       ${PATCH_SIZE}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Stain norm:       ${STAIN_NORM:-disabled}"
echo "Device:           ${DEVICE}"
echo "=============================================="

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Count SVS files
SVS_COUNT=$(find "${INPUT_DIR}" -name "*.svs" | wc -l)
echo "Found ${SVS_COUNT} SVS files"

if [ "${SVS_COUNT}" -eq 0 ]; then
    echo "Error: No SVS files found in ${INPUT_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check Python environment
echo ""
echo "Checking Python environment..."
python3 -c "import openslide; print(f'OpenSlide version: {openslide.__version__}')" 2>/dev/null || {
    echo "Warning: OpenSlide not installed. Install with:"
    echo "  pip install openslide-python"
    echo "  apt-get install openslide-tools"
}

python3 -c "import timm; print(f'TIMM version: {timm.__version__}')" 2>/dev/null || {
    echo "Warning: TIMM not installed. Install with: pip install timm"
}

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "Error: PyTorch not installed"
    exit 1
}

# Run preprocessing
echo ""
echo "Starting preprocessing..."
echo ""

cd "${PROJECT_ROOT}"

python3 -m wsi_model.src.preprocessing.wsi_preprocessor \
    "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --model "${MODEL_NAME}" \
    --patch-size "${PATCH_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    ${STAIN_NORM} \
    --device "${DEVICE}"

echo ""
echo "=============================================="
echo "Preprocessing completed!"
echo "Features saved to: ${OUTPUT_DIR}/features/"
echo "=============================================="
