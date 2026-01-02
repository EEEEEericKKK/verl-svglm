#!/bin/bash

# Example: Run inference on Geometry3k test set
# This script shows different usage scenarios

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================================="
echo "Example Usage Scenarios for vLLM Multi-turn Inference"
echo "=========================================================================="
echo ""

# ============================================================================
# Example 1: Basic inference on test set (10 samples)
# ============================================================================

echo "Example 1: Basic inference on test set (10 samples)"
echo "----------------------------------------------------------------------"

# export MODEL_PATH="/proj/inf-scaling/csl/svglm/checkpoints/verl_sft_geo3k/geo3k-qwen3-vl-8b_v1_merged/global_step_1000"
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATA_PATH="/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_eval/test.parquet"
export IMAGE_FOLDER="/proj/inf-scaling/csl/svglm/data/geo3k_toolcall"
export OUTPUT_PATH="/proj/inf-scaling/csl/svglm/verl/outputs/inference/geo3k_zeroshot_instruct_origin.jsonl"
export NUM_SAMPLES=""
export TENSOR_PARALLEL_SIZE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MAX_TURNS=10
export TEMPERATURE=0.7

bash "$SCRIPT_DIR/run_vllm_multiturn_inference.sh"

echo ""
echo ""

# ============================================================================
# Example 2: Full test set with multiple GPUs
# ============================================================================

# Uncomment to run:
# echo "Example 2: Full test set with 4 GPUs (tensor parallel)"
# echo "----------------------------------------------------------------------"
# 
# export MODEL_PATH="/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking"
# export DATA_PATH="/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_w_tool/test.parquet"
# export IMAGE_FOLDER="/proj/inf-scaling/csl/svglm/data/geo3k_toolcall"
# export OUTPUT_PATH="/proj/inf-scaling/csl/svglm/verl/outputs/inference/geo3k_test_full.jsonl"
# export NUM_SAMPLES=""  # Process all samples
# export TENSOR_PARALLEL_SIZE=4
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export MAX_TURNS=15
# export TEMPERATURE=0.7
# 
# bash "$SCRIPT_DIR/run_vllm_multiturn_inference.sh"

# ============================================================================
# Example 3: Low temperature for deterministic generation
# ============================================================================

# Uncomment to run:
# echo "Example 3: Deterministic generation (low temperature)"
# echo "----------------------------------------------------------------------"
# 
# export MODEL_PATH="/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking"
# export DATA_PATH="/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_w_tool/test.parquet"
# export IMAGE_FOLDER="/proj/inf-scaling/csl/svglm/data/geo3k_toolcall"
# export OUTPUT_PATH="/proj/inf-scaling/csl/svglm/verl/outputs/inference/geo3k_test_deterministic.jsonl"
# export NUM_SAMPLES=50
# export TEMPERATURE=0.1
# export TOP_P=0.95
# export MAX_TURNS=10
# 
# bash "$SCRIPT_DIR/run_vllm_multiturn_inference.sh"

# ============================================================================
# Example 4: Train set sampling
# ============================================================================

# Uncomment to run:
# echo "Example 4: Train set sampling"
# echo "----------------------------------------------------------------------"
# 
# export MODEL_PATH="/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking"
# export DATA_PATH="/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_w_tool/train.parquet"
# export IMAGE_FOLDER="/proj/inf-scaling/csl/svglm/data/geo3k_toolcall"
# export OUTPUT_PATH="/proj/inf-scaling/csl/svglm/verl/outputs/inference/geo3k_train_sample.jsonl"
# export NUM_SAMPLES=100
# export TEMPERATURE=0.8
# export MAX_TURNS=10
# 
# bash "$SCRIPT_DIR/run_vllm_multiturn_inference.sh"

echo ""
echo "=========================================================================="
echo "Examples completed!"
echo "=========================================================================="
