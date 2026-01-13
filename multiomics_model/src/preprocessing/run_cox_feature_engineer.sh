#!/bin/bash

# TCGA PANCAN Cox Regression Analysis - Background Execution Script
# =================================================================
#
# This script runs the comprehensive Cox regression analysis in the background
# with optimized parallel processing settings.
#
# Usage:
#   cd src/preprocessing
#   chmod +x run_cox_feature_engineer.sh
#   ./run_cox_feature_engineer.sh
#
# Monitor progress:
#   tail -f cox_analysis.log
#
# Stop execution:
#   ps aux | grep cox_feature_engineer
#   kill <PID>

echo "🚀 Starting TCGA PANCAN Cox Regression Analysis..."
echo "📅 Started at: $(date)"

# Configuration
DATA_DIR="../../data/raw"
OUTPUT_DIR="../../data/processed"
RESULTS_DIR="../../results"
MAX_WORKERS=3                # Number of cancer types processed in parallel
MIN_PATIENTS=20             # Minimum patients per cancer type
P_THRESHOLD=0.05            # P-value significance threshold
LOG_FILE="cox_analysis_$(date +%Y%m%d_%H%M%S).log"

# GPU Settings
USE_GPU=false               # Set to true to enable GPU acceleration
                           # Note: GPU was slower in previous tests

# Additional options
SKIP_VISUALIZATION=true     # Skip plots for faster background execution

# Build command
CMD="python cox_feature_engineer.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --max-workers $MAX_WORKERS"
CMD="$CMD --min-patients $MIN_PATIENTS"
CMD="$CMD --p-threshold $P_THRESHOLD"
CMD="$CMD --log-file $LOG_FILE"

if [ "$USE_GPU" = true ]; then
    CMD="$CMD --use-gpu"
    echo "🖥️  GPU acceleration: ENABLED"
else
    echo "🖥️  GPU acceleration: DISABLED (CPU multiprocessing)"
fi

if [ "$SKIP_VISUALIZATION" = true ]; then
    CMD="$CMD --skip-visualization"
    echo "📊 Visualization: SKIPPED (faster execution)"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Max workers (cancer-level): $MAX_WORKERS"
echo "  Min patients per cancer: $MIN_PATIENTS"
echo "  P-value threshold: $P_THRESHOLD"
echo "  Log file: $LOG_FILE"
echo ""
echo "Parallel processing structure:"
echo "  • Cancer-level: $MAX_WORKERS parallel processes"
echo "  • Feature-level: 6 CPU processes per cancer"
echo "  • Total CPU processes: $(($MAX_WORKERS * 6))"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' not found!"
    echo "Please check the path and try again."
    exit 1
fi

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# Start the analysis in background
echo "🔥 Starting analysis in background..."
echo "Command: $CMD"
echo ""

# Run in background with nohup
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ Analysis started successfully!"
echo "📊 Process ID: $PID"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps aux | grep cox_feature_engineer"
echo ""
echo "Stop execution:"
echo "  kill $PID"
echo ""
echo "Estimated completion time: 1-2 hours"
echo "🎯 The analysis will process 5 omics types across multiple cancer types."
echo ""