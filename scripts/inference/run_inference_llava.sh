#!/bin/bash
# Multi-turn VLM Inference Runner
# This script runs multi-modal, multi-turn inference with vLLM

set -e  # Exit on error

export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/proj/inf-scaling/huggingface
export TMPDIR=/proj/inf-scaling/TMP
export VLLM_CACHE_ROOT=/proj/inf-scaling/csl/.cache

export WANDB_API_KEY=55e59d4db1f11a22713ac08a884b1b44ce20caf2
export WANDB_PROJECT=llamafactory-mathcanvas-sft
export WANDB_NAME=mathcanvas-llava3-8b-full-sft

# ============================================================================
# Default Configuration
# ============================================================================

# Model configuration
MODEL_PATH="${MODEL_PATH:-/proj/inf-scaling/csl/svglm/LlamaFactory/saves/llava-v1.6-mistral-7b-hf/full/sft_mathcanvas}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_GPUS="${NUM_GPUS:-8}"  # Number of GPUs for data parallelism

# Data configuration
DATA_PATH="${DATA_PATH:-/proj/inf-scaling/csl/svglm/data/hf_template/mathcanvas_test/mathcanvas_test.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-/proj/inf-scaling/csl/svglm/verl/outputs/inference/Mathcanvas-llava-sft.jsonl}"
MESSAGES_KEY="${MESSAGES_KEY:-messages}"
IMAGE_KEY="${IMAGE_KEY:-question_images}"
VIDEO_KEY="${VIDEO_KEY:-videos}"
IMAGE_PATCH_SIZE="${IMAGE_PATCH_SIZE:-14}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
GROUND_TRUTH_KEY="${GROUND_TRUTH_KEY:-answer}"
KEEP_ASSISTANT="${KEEP_ASSISTANT:-first}"  # "first" or "last" - which assistant message to keep before generation
DATASET_FORMAT="${DATASET_FORMAT:-mathcanvas}"  # Dataset format: "sft" or custom formats

# Generation configuration
MAX_TURNS="${MAX_TURNS:-3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
ENABLE_TOOL_CALL="${ENABLE_TOOL_CALL:-true}"


# ============================================================================
# Run Inference
# ============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "Multi-turn VLM Inference"
echo "=================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "Max turns: $MAX_TURNS"
echo "Temperature: $TEMPERATURE"
echo "Tool calls: $ENABLE_TOOL_CALL"
echo "Ground truth key: $GROUND_TRUTH_KEY"
echo "Keep assistant: $KEEP_ASSISTANT"
echo "Dataset format: $DATASET_FORMAT"
echo "Num GPUs: $NUM_GPUS"
echo "=================================="
echo ""

# Build python command
CMD="python ${SCRIPT_DIR}/multiturn_vllm_inference.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"
CMD="$CMD --max_model_len $MAX_MODEL_LEN"
CMD="$CMD --max_num_seqs $MAX_NUM_SEQS"
CMD="$CMD --num_gpus $NUM_GPUS"
CMD="$CMD --data_path $DATA_PATH"
CMD="$CMD --output_path $OUTPUT_PATH"
CMD="$CMD --messages_key $MESSAGES_KEY"
CMD="$CMD --image_key $IMAGE_KEY"
CMD="$CMD --video_key $VIDEO_KEY"
CMD="$CMD --image_patch_size $IMAGE_PATCH_SIZE"
CMD="$CMD --max_turns $MAX_TURNS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --enable_prefix_caching"
CMD="$CMD --ground_truth_key $GROUND_TRUTH_KEY"
CMD="$CMD --keep_assistant $KEEP_ASSISTANT"
CMD="$CMD --dataset_format $DATASET_FORMAT"

if [ "$ENABLE_TOOL_CALL" = "true" ]; then
    CMD="$CMD --enable_tool_call"
fi

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Execute
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=================================="
echo "Inference complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "=================================="
