#!/usr/bin/env bash
set -xeuo pipefail

# =================================================================================
# MULTINODE ENVIRONMENT DISCOVERY FROM LSF
# =================================================================================

# Get a CLEAN, SPACE-SEPARATED list of short hostnames from LSF.
CLEAN_HOST_LIST=$(echo "$LSB_MCPU_HOSTS" | awk '{for(i=1; i<=NF; i+=2) printf "%s ", $i}')

# Get the total number of nodes (world size) from the clean list.
export NNODES=$(echo "$CLEAN_HOST_LIST" | wc -w)

# Get the master node's address (the first host in the clean list).
export MASTER_ADDR=$(echo "$CLEAN_HOST_LIST" | awk '{print $1}')

export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Get the SHORT hostname of the current machine ('-s' flag is important).
CURRENT_HOST=$(hostname -s)

# Find the rank of the current node in the clean list.
HOST_ARRAY=($CLEAN_HOST_LIST)
export NODE_RANK=-1
for i in "${!HOST_ARRAY[@]}"; do
   if [[ "${HOST_ARRAY[$i]}" == "$CURRENT_HOST" ]]; then
       export NODE_RANK=$i
       break
   fi
done

# Error out if the rank could not be determined (in multinode setup).
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" -eq -1 ]; then
    echo "ERROR: Could not determine NODE_RANK." >&2
    echo "  - Current (short) Host: $CURRENT_HOST" >&2
    echo "  - Clean Host List Searched: ${CLEAN_HOST_LIST}" >&2
    echo "  - Original LSF Host List: $LSB_MCPU_HOSTS" >&2
    exit 1
fi

# Define a static port for the master node to listen on.
export MASTER_PORT=${MASTER_PORT:-29500}

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

eval "$(conda shell.bash hook)"

conda activate /proj/inf-scaling/csl/myconda/verl-svglm

export WANDB_API_KEY=55e59d4db1f11a22713ac08a884b1b44ce20caf2
export WANDB_PROJECT=verl-mathcanvas-rlvr
export WANDB_NAME=initial

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export RAY_TMPDIR=/proj/inf-scaling/csl/svglm/tmp_ray

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Thinking \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=/proj/inf-scaling/csl/svglm/data/mathcanvas_multiturn_rl/train.parquet \
    data.val_files=/proj/inf-scaling/csl/svglm/data/mathcanvas_multiturn_rl/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/svg_to_png_tool_config.yaml" \
    trainer.total_epochs=15 $@

