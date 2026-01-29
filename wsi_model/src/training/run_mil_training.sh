#!/bin/bash
# =============================================================================
# MIL Model Training Script
# =============================================================================
#
# Trains ABMIL or TransMIL models for WSI survival prediction.
#
# Usage:
#   ./run_mil_training.sh [OPTIONS]
#
# Options:
#   --model TYPE        Model type: abmil or transmil (default: abmil)
#   --features DIR      Features directory (default: ../../data/processed/features)
#   --labels FILE       Labels CSV file (required)
#   --output DIR        Output directory (default: ../../results)
#   --epochs N          Number of epochs (default: 100)
#   --batch_size N      Batch size (default: 1)
#   --lr RATE           Learning rate (default: 1e-4)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default parameters
FEATURES_DIR="${PROJECT_ROOT}/wsi_model/data/processed/features"
OUTPUT_DIR="${PROJECT_ROOT}/wsi_model/results"
MODEL="abmil"
EPOCHS=100
BATCH_SIZE=1
LR=1e-4
MAX_PATCHES=10000
DEVICE="cuda:0"

# Labels path (needs to be provided or created)
LABELS_PATH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --features)
            FEATURES_DIR="$2"
            shift 2
            ;;
        --labels)
            LABELS_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max_patches)
            MAX_PATCHES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model TYPE        Model type: abmil or transmil (default: abmil)"
            echo "  --features DIR      Features directory"
            echo "  --labels FILE       Labels CSV file"
            echo "  --output DIR        Output directory"
            echo "  --epochs N          Number of epochs (default: 100)"
            echo "  --batch_size N      Batch size (default: 1)"
            echo "  --lr RATE           Learning rate (default: 1e-4)"
            echo "  --max_patches N     Max patches per slide (default: 10000)"
            echo "  --device DEVICE     CUDA device (default: cuda:0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check labels path
if [ -z "${LABELS_PATH}" ]; then
    echo "Error: --labels is required"
    echo "Run with --help for usage"
    exit 1
fi

if [ ! -f "${LABELS_PATH}" ]; then
    echo "Error: Labels file not found: ${LABELS_PATH}"
    exit 1
fi

# Check features directory
if [ ! -d "${FEATURES_DIR}" ]; then
    echo "Error: Features directory not found: ${FEATURES_DIR}"
    exit 1
fi

# Count feature files
FEATURE_COUNT=$(ls -1 "${FEATURES_DIR}"/*.h5 2>/dev/null | wc -l)
echo "Found ${FEATURE_COUNT} feature files"

if [ "${FEATURE_COUNT}" -eq 0 ]; then
    echo "Error: No HDF5 feature files found in ${FEATURES_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Print configuration
echo "=============================================="
echo "MIL Model Training"
echo "=============================================="
echo "Model:            ${MODEL}"
echo "Features:         ${FEATURES_DIR}"
echo "Labels:           ${LABELS_PATH}"
echo "Output:           ${OUTPUT_DIR}"
echo "Epochs:           ${EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Learning rate:    ${LR}"
echo "Max patches:      ${MAX_PATCHES}"
echo "Device:           ${DEVICE}"
echo "Feature files:    ${FEATURE_COUNT}"
echo "=============================================="

# Run training
cd "${PROJECT_ROOT}"

python -m wsi_model.src.training.train_mil \
    --features_dir "${FEATURES_DIR}" \
    --labels "${LABELS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --max_patches "${MAX_PATCHES}" \
    --device "${DEVICE}"

echo ""
echo "=============================================="
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
