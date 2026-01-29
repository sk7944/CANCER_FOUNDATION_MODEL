#!/bin/bash
# =============================================================================
# WSI Preprocessing Pipeline - Background Runner
# =============================================================================
#
# This script runs the WSI preprocessing pipeline in the background,
# with proper logging and persistence after session exit.
#
# Usage:
#   nohup ./run_preprocessing_background.sh > /dev/null 2>&1 &
#
# Or simply:
#   ./run_preprocessing_background.sh
#
# Logs are saved to: wsi_model/data/processed/logs/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Configuration
INPUT_DIR="${PROJECT_ROOT}/wsi_model/data/raw"
OUTPUT_DIR="${PROJECT_ROOT}/wsi_model/data/processed"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/preprocessing_${TIMESTAMP}.log"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}/features"

# Start logging
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "WSI Preprocessing Pipeline - Background Mode"
echo "=============================================="
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

# Count total SVS files
TOTAL_FILES=$(find "${INPUT_DIR}" -name "*.svs" | wc -l)
echo "Total SVS files to process: ${TOTAL_FILES}"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import openslide; print(f'OpenSlide: OK ({openslide.__version__})')" 2>/dev/null || {
    echo "ERROR: OpenSlide not installed"
    echo "Install with: pip install openslide-python && apt-get install openslide-tools"
    exit 1
}

python3 -c "import timm; print(f'TIMM: OK ({timm.__version__})')" 2>/dev/null || {
    echo "ERROR: TIMM not installed"
    echo "Install with: pip install timm"
    exit 1
}

python3 -c "import torch; print(f'PyTorch: OK ({torch.__version__}), CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Starting preprocessing..."
echo ""

# Run preprocessing with Python
cd "${PROJECT_ROOT}"

python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, '/data4/workspace_vscode/CANCER_FOUNDATION_MODEL')

from wsi_model.src.preprocessing import WSIPreprocessor
from wsi_model.src.preprocessing.wsi_preprocessor import PreprocessingConfig

# Configuration
INPUT_DIR = '/data4/workspace_vscode/CANCER_FOUNDATION_MODEL/wsi_model/data/raw'
OUTPUT_DIR = '/data4/workspace_vscode/CANCER_FOUNDATION_MODEL/wsi_model/data/processed'

config = PreprocessingConfig(
    patch_size=256,
    patch_level=0,
    min_tissue_ratio=0.8,
    blur_threshold=50.0,
    stain_normalize=True,
    stain_augment=False,
    model_name='swin_tiny',
    batch_size=64,
    num_workers=4,
    save_patches=False,
    save_features=True,
)

print(f"Configuration:")
print(f"  Patch size: {config.patch_size}")
print(f"  Model: {config.model_name}")
print(f"  Stain normalize: {config.stain_normalize}")
print(f"  Batch size: {config.batch_size}")
print("")

# Create preprocessor
preprocessor = WSIPreprocessor(
    output_dir=OUTPUT_DIR,
    config=config,
    device='cuda:0',
)

# Get list of cancer types
cancer_types = sorted([
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
    and d not in ['gdc_manifest', '.']
    and not d.endswith('.log')
    and not d.endswith('.txt')
    and not d.endswith('.tsv')
    and not d.endswith('.xlsx')
])

print(f"Found {len(cancer_types)} cancer types: {cancer_types}")
print("")

# Track progress
all_results = []
start_time = time.time()

for i, cancer in enumerate(cancer_types):
    cancer_dir = os.path.join(INPUT_DIR, cancer)

    # Count files in this cancer type
    svs_files = list(Path(cancer_dir).rglob('*.svs'))

    if not svs_files:
        print(f"[{i+1}/{len(cancer_types)}] {cancer}: No SVS files, skipping")
        continue

    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(cancer_types)}] Processing {cancer}: {len(svs_files)} files")
    print(f"{'='*60}")

    cancer_start = time.time()

    results = preprocessor.process_directory(
        input_dir=cancer_dir,
        pattern='*.svs',
        recursive=True,
    )

    all_results.extend(results)

    # Print cancer type summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_patches = sum(r.num_patches_valid for r in results if r.success)
    cancer_time = time.time() - cancer_start

    print(f"\n{cancer} Summary:")
    print(f"  Processed: {len(results)} slides")
    print(f"  Successful: {successful}, Failed: {failed}")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Time: {cancer_time/60:.1f} minutes")

    # Save intermediate progress
    progress_file = os.path.join(OUTPUT_DIR, 'logs', 'progress.json')
    progress = {
        'current_cancer': cancer,
        'processed_cancers': i + 1,
        'total_cancers': len(cancer_types),
        'total_slides_processed': len(all_results),
        'successful_slides': sum(1 for r in all_results if r.success),
        'last_update': datetime.now().isoformat(),
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

# Final summary
total_time = time.time() - start_time
summary = preprocessor.get_summary(all_results)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Total slides processed: {summary['total_slides']}")
print(f"Successful: {summary['successful']}")
print(f"Failed: {summary['failed']}")
print(f"Total patches extracted: {summary['total_patches']:,}")
print(f"Average patches per slide: {summary['avg_patches_per_slide']:.0f}")
print(f"Total time: {total_time/3600:.1f} hours")
print(f"Average time per slide: {summary['avg_time_per_slide_seconds']:.1f} seconds")

if summary['failed_slides']:
    print(f"\nFailed slides ({len(summary['failed_slides'])}):")
    for slide_id in summary['failed_slides'][:10]:
        print(f"  - {slide_id}")
    if len(summary['failed_slides']) > 10:
        print(f"  ... and {len(summary['failed_slides']) - 10} more")

print(f"\nCompleted at: {datetime.now().isoformat()}")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Preprocessing completed at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "=============================================="
