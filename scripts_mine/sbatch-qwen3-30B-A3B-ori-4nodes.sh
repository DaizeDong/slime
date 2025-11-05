#!/bin/bash
#SBATCH --job-name=ori-4nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --reservation=moe
#SBATCH --output=/mnt/weka/home/haolong.jia/workspace/rlhf/slime/logs/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/haolong.jia/workspace/rlhf/slime/logs/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

TIME=$(date +%Y%m%d-%H%M%S)
LOGROOT="/mnt/weka/home/haolong.jia/workspace/rlhf/slime/logs/sbatch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOGROOT"

# redirect stdout and stderr to separate files with tee
log_file="${TIME}.${SLURMD_NODENAME}.${SLURM_JOB_ID}.${SLURM_PROCID}.log"
err_file="${TIME}.${SLURMD_NODENAME}.${SLURM_JOB_ID}.${SLURM_PROCID}.err"
exec > >(tee -a "$LOGROOT/${log_file}") \
     2> >(tee -a "$LOGROOT/${err_file}" >&2)

echo "========== SLURM ENV DUMP =========="
env | grep '^SLURM' | sort
echo "===================================="

set -euo pipefail

################################################################################
HEAD_NODE="$(echo "$SLURM_NODELIST" | sed 's/[][]//g' | cut -d, -f1)"
export HEAD_NODE

export RAY_PORT="${RAY_PORT:-6379}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
echo "[INFO] RAY_PORT=$RAY_PORT"
echo "[INFO] DASHBOARD_PORT=$DASHBOARD_PORT"

################################################################################
#SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SCRIPT_DIR=/root/slime/scripts_mine

export model_name="Qwen3-30B-A3B"
export run_postfix="${SLURM_JOB_NAME}"
export wandb_key_file="scripts_mine/wandb_key_sandy.txt"

export hf_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint/${model_name}"
export dist_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint_torch_dist/${model_name}"
export megatron_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint_megatron/${model_name}-${run_postfix}"
export start_rollout_id=$(val=$(cat "${megatron_ckpt_path}/latest_checkpointed_iteration.txt" 2>/dev/null || true); if [[ "$val" =~ ^[0-9]+$ ]]; then echo $((val+1)); else echo ""; fi)

export prompt_data="/mnt/sharefs/users/haolong.jia/RL-data/dapo-math-17k/dapo-math-17k.jsonl"
export eval_prompt_data="aime /mnt/sharefs/users/haolong.jia/RL-data/aime-2024/aime-2024.jsonl"
################################################################################

# container
CONTAINER_IMAGE="slimerl/slime:latest"
CONTAINER_MOUNTS="/mnt/sharefs/users/haolong.jia:/mnt/sharefs/users/haolong.jia:rw,/mnt/weka/home/haolong.jia/workspace/rlhf/slime:/root/slime:rw,/mnt/weka/home/haolong.jia/workspace/rlhf/Megatron-LM:/root/out-Megatron-LM:rw,/mnt/weka/home/haolong.jia/workspace/rlhf/sglang:/root/out-sglang:rw"

srun \
  --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  bash -lc '
set -euo pipefail

echo "[NODE $(hostname -s)] starting bootstrap..."

# in case of early exit, do cleanup
trap "echo [NODE $(hostname -s)] cleanup; ray stop --force >/dev/null 2>&1 || true; pkill -9 python >/dev/null 2>&1 || true" EXIT

# clean up old processes
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
pkill -9 redis 2>/dev/null || true

# python environment
cd /root/slime
pip install -e .
cp -rf /root/out-Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py /root/Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py
cp -rf /root/out-Megatron-LM/megatron/training/checkpointing.py /root/Megatron-LM/megatron/training/checkpointing.py
#cp -rf /root/out-Megatron-LM/megatron/training/arguments.py /root/Megatron-LM/megatron/training/arguments.py
#cp -rf /root/out-Megatron-LM/megatron/training/global_vars.py /root/Megatron-LM/megatron/training/global_vars.py

# running environment
export PYTHONBUFFERED=16
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# training environment
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"
export WANDB_KEY="$(cat ${wandb_key_file})"
export WANDB_GROUP=${model_name}-${run_postfix}

# model args
CKPT_ARGS=(
  --hf-checkpoint ${hf_ckpt_path}
  --ref-load ${dist_ckpt_path}
  --load ${megatron_ckpt_path}
  --save ${megatron_ckpt_path}
  --save-interval 50
)

if [ -n "$start_rollout_id" ]; then
  CKPT_ARGS+=(--start-rollout-id "$start_rollout_id") # ------ to continue from previous run ------
fi

ROLLOUT_ARGS=(
  --prompt-data ${prompt_data}
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout 1000
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 8192
  --rollout-temperature 0.8
  --global-batch-size 256
  --balance-data
)

EVAL_ARGS=(
  --eval-interval 50
  --eval-prompt-data ${eval_prompt_data}
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 0.7
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98

  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
  --use-wandb
  --wandb-project slime
  --wandb-group ${WANDB_GROUP}
  --wandb-key ${WANDB_KEY}
  --disable-wandb-random-suffix
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 8
  --sglang-mem-fraction-static 0.6
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
  # default dropout in megatron is 0.1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  # should be good for model performance
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # need to comment this when using model with MLA
  --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Start Ray cluster
THIS_NODE="$(hostname -s)"
if [ "$THIS_NODE" = "$HEAD_NODE" ]; then
  echo "[HEAD] starting ray head on ${HEAD_NODE}:${RAY_PORT} ..."
  ray start --head \
    --node-ip-address "$HEAD_NODE" \
    --port "$RAY_PORT" \
    --num-gpus 8 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port "$DASHBOARD_PORT" \
    # --disable-usage-stats

  sleep 10

  echo "[HEAD] submitting ray job ..."
  ray job submit --address="http://${HEAD_NODE}:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
      --actor-num-nodes "'"${SLURM_NNODES}"'" \
      --actor-num-gpus-per-node 8 \
      --colocate \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${GRPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]}

  echo "[HEAD] job finished, stopping ray..."

else
  echo "[WORKER $(hostname -s)] waiting for ${HEAD_NODE}:${RAY_PORT} ..."
  for i in $(seq 1 120); do
    (echo > /dev/tcp/"$HEAD_NODE"/"$RAY_PORT") >/dev/null 2>&1 && break
    sleep 1
  done

  echo "[WORKER $(hostname -s)] joining cluster..."
  ray start --address "$HEAD_NODE:$RAY_PORT" --num-gpus 8 --disable-usage-stats

  sleep infinity
fi
'
