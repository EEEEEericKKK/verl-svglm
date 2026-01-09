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


# =================================================================================

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

TRAIN_FILES=/proj/inf-scaling/csl/svglm/data/geo3k_toolcall/processed_data_verl.parquet

backend=${BACKEND:-fsdp}

project_name=verl_sft_geo3k

RESUME_MODE=auto
MODEL_ID=/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Instruct
# MODEL_ID=${HDFS_ROOT}/model/Qwen3-VL-30B-A3B-Instruct

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:--1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=2e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=geo3k-qwen3-vl-8b-instruct-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp-1202a1
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=geo3k-qwen3-vl-8b-instruct-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-megatron-1202a1
fi
exp_name=geo3k-qwen3-vl-8b-instruct_v1

CKPT_HOME="/proj/inf-scaling/csl/svglm/checkpoints/${project_name}/${exp_name}"
mkdir -p "${CKPT_HOME}"

export WANDB_API_KEY=55e59d4db1f11a22713ac08a884b1b44ce20caf2
export WANDB_PROJECT=verl-geo3k-sft
export WANDB_NAME=initial

torchrun --nnodes=$NNODES --nproc-per-node=$NGPUS_PER_NODE --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=8 \
    data.max_length=131072 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=131072 \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=after_each_epoch \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=3 \
    checkpoint.save_contents=[model]