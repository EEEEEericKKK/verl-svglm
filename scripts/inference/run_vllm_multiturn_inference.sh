#!/bin/bash

# Multi-modal, Multi-turn Tool Use Inference with vLLM
# Run inference on Geometry3k dataset with SVG to PNG tool

set -e

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_PATH="${MODEL_PATH:-/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# Data configuration
DATA_PATH="${DATA_PATH:-/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_rl/test.parquet}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/proj/inf-scaling/csl/svglm/data/geo3k_toolcall}"
OUTPUT_PATH="${OUTPUT_PATH:-/proj/inf-scaling/csl/svglm/verl/outputs/inference/geo3k_multiturn_inference.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

# Generation configuration
MAX_TURNS="${MAX_TURNS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16384}"

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# Print Configuration
# ============================================================================

echo "=========================================================================="
echo "vLLM Multi-turn Inference Configuration"
echo "=========================================================================="
echo "Model Configuration:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS: $MAX_NUM_SEQS"
echo ""
echo "Data Configuration:"
echo "  DATA_PATH: $DATA_PATH"
echo "  IMAGE_FOLDER: $IMAGE_FOLDER"
echo "  OUTPUT_PATH: $OUTPUT_PATH"
echo "  NUM_SAMPLES: ${NUM_SAMPLES:-all}"
echo ""
echo "Generation Configuration:"
echo "  MAX_TURNS: $MAX_TURNS"
echo "  TEMPERATURE: $TEMPERATURE"
echo "  TOP_P: $TOP_P"
echo "  MAX_TOKENS: $MAX_TOKENS"
echo ""
echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="
echo ""

# ============================================================================
# Check dependencies
# ============================================================================

echo "Checking dependencies..."

# Check if Python packages are installed
python -c "import vllm" 2>/dev/null || {
    echo "Error: vllm is not installed. Please install it with: pip install vllm"
    exit 1
}

python -c "import cairosvg" 2>/dev/null || {
    echo "Error: cairosvg is not installed. Please install it with: pip install cairosvg"
    exit 1
}

python -c "import qwen_vl_utils" 2>/dev/null || {
    echo "Error: qwen_vl_utils is not installed. Please install it with: pip install qwen-vl-utils"
    exit 1
}

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file does not exist: $DATA_PATH"
    exit 1
fi

echo "All dependencies checked successfully."
echo ""

# ============================================================================
# Create output directory
# ============================================================================

OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Run inference
# ============================================================================

echo "Starting inference..."
echo ""

# Build command
CMD="python $(dirname $0)/vllm_multiturn_inference.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"
CMD="$CMD --max_model_len $MAX_MODEL_LEN"
CMD="$CMD --max_num_seqs $MAX_NUM_SEQS"
CMD="$CMD --data_path $DATA_PATH"
CMD="$CMD --output_path $OUTPUT_PATH"
CMD="$CMD --max_turns $MAX_TURNS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --max_tokens $MAX_TOKENS"

# Add optional arguments
if [ -n "$IMAGE_FOLDER" ]; then
    CMD="$CMD --image_folder $IMAGE_FOLDER"
fi

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Add prefix caching flag
CMD="$CMD --enable_prefix_caching"

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "=========================================================================="
echo "Inference completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================================================="
