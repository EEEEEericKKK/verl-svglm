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

export MODEL_PATH="/proj/inf-scaling/csl/svglm/checkpoints/verl_sft_mathcanvas/mathcanvas-qwen3-vl-8b-instruct_v1_merged/global_step_964"
# export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export DATA_PATH="/proj/inf-scaling/csl/svglm/data/mathcanvas_test/mathcanvas_test.parquet"
export IMAGE_FOLDER="/proj/inf-scaling/csl/svglm/data/mathcanvas_toolcall"
export OUTPUT_PATH="/proj/inf-scaling/csl/svglm/verl/outputs/inference/mathcanvas_sft_training_split.jsonl"
export NUM_SAMPLES=""
export TENSOR_PARALLEL_SIZE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MAX_TURNS=10
export TEMPERATURE=0.7

bash "$SCRIPT_DIR/run_vllm_multiturn_inference.sh"

echo ""
echo "=========================================================================="
echo "Examples completed!"
echo "=========================================================================="
