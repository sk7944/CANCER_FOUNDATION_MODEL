#!/bin/bash

# TCGA PANCAN Integrated Dataset Creation - Background Execution Script
# =====================================================================
#
# This script runs the comprehensive integrated dataset creation process
# in the background for TabTransformer model preparation.
#
# Usage:
#   cd src/preprocessing
#   chmod +x run_integrated_dataset_builder.sh
#   ./run_integrated_dataset_builder.sh
#
# Monitor progress:
#   tail -f integrated_dataset.log
#
# Stop execution:
#   ps aux | grep integrated_dataset_builder
#   kill <PID>

echo "🚀 Starting TCGA PANCAN Integrated Dataset Creation..."
echo "📅 Started at: $(date)"

# Configuration
DATA_DIR="../../data/processed"      # Directory with processed omics data
OUTPUT_DIR="../../data/processed"    # Output directory for integrated datasets
RESULTS_DIR="../../results"          # Results and summary directory
MAX_FEATURES=0                   # 0=unlimited, 모든 features 사용
TRAIN_RATIO=0.7                  # Training set ratio
VAL_RATIO=0.15                   # Validation set ratio
TEST_RATIO=0.15                  # Test set ratio
RANDOM_SEED=42                   # Random seed for reproducibility
LOG_FILE="integrated_dataset_$(date +%Y%m%d_%H%M%S).log"

# Build command
CMD="python integrated_dataset_builder.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --max-features-per-omics $MAX_FEATURES"
CMD="$CMD --train-ratio $TRAIN_RATIO"
CMD="$CMD --val-ratio $VAL_RATIO"
CMD="$CMD --test-ratio $TEST_RATIO"
CMD="$CMD --random-seed $RANDOM_SEED"
CMD="$CMD --log-file $LOG_FILE"

# Display configuration
echo ""
echo "Configuration:"
echo "  Data directory (processed omics): $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Max features per omics type: $MAX_FEATURES"
echo "  Data split ratios: Train=$TRAIN_RATIO, Val=$VAL_RATIO, Test=$TEST_RATIO"
echo "  Random seed: $RANDOM_SEED"
echo "  Log file: $LOG_FILE"
echo ""
echo "Processing pipeline:"
echo "  1. Load Cox regression coefficients and processed omics data"
echo "  2. Create integrated Cox table (Expression + CNV + microRNA + RPPA + Mutations)"
echo "  3. Create methylation table (separate processing)"
echo "  4. Generate PyTorch Dataset classes"
echo "  5. Create stratified train/val/test splits"
echo "  6. Save all outputs for TabTransformer training"
echo ""

# Check if processed data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Processed data directory '$DATA_DIR' not found!"
    echo "Please ensure Cox feature engineering has been completed first."
    echo "Run './run_cox_analysis.sh' to generate the required processed data."
    exit 1
fi

# Check for required Cox coefficient files
REQUIRED_FILES=(
    "cox_coefficients_expression.parquet"
    "cox_coefficients_cnv.parquet"
    "cox_coefficients_microrna.parquet"
    "cox_coefficients_rppa.parquet"
    "cox_coefficients_mutations.parquet"
    "processed_expression_data.parquet"
    "processed_cnv_data.parquet"
    "processed_microrna_data.parquet"
    "processed_rppa_data.parquet"
    "processed_mutations_data.parquet"
    "methylation_data_for_tabtransformer.parquet"
    "processed_clinical_data.parquet"
    "processed_clinical_data_for_methylation.parquet"  # 8,224명 전체 임상데이터
)

echo "Checking for required input files..."
MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Error: Missing required processed data files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "    • $DATA_DIR/$file"
    done
    echo ""
    echo "Please run Cox feature engineering first:"
    echo "    ./run_cox_feature_engineer.sh"
    exit 1
fi

echo "✅ All required input files found!"
echo ""

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# Estimate disk space requirements
echo "Estimating resource requirements..."
INPUT_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
AVAILABLE_SPACE=$(df -h "$OUTPUT_DIR" | tail -1 | awk '{print $4}')
echo "  Input data size: $INPUT_SIZE"
echo "  Available disk space: $AVAILABLE_SPACE"
echo "  Estimated output size: ~2-3GB (integrated tables)"
echo ""

# Start the analysis in background
echo "🔥 Starting integrated dataset creation in background..."
echo "Command: $CMD"
echo ""

# Run in background with nohup
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ Dataset creation started successfully!"
echo "📊 Process ID: $PID"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps aux | grep integrated_dataset_builder"
echo ""
echo "Stop execution:"
echo "  kill $PID"
echo ""
echo "Expected outputs:"
echo "  📊 $OUTPUT_DIR/integrated_table_cox.parquet"
echo "  📊 $OUTPUT_DIR/methylation_table.parquet"
echo "  📊 $OUTPUT_DIR/train_val_test_splits.json"
echo "  📈 $RESULTS_DIR/integrated_dataset_summary.json"
echo "  📈 $RESULTS_DIR/integrated_dataset_validation.json"
echo ""
echo "Estimated completion time: 10-30 minutes"
echo "🎯 This process will create ready-to-use datasets for TabTransformer training."
echo ""