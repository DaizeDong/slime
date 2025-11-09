#!/bin/bash
#SBATCH --job-name=tuned-off4-replay-union-2nodes-async
#SBATCH --nodes=2
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

exec 3>&1 4>&2
trap 'exec 1>&3 2>&4 3>&- 4>&-' EXIT

exec > >(tee -a "$LOGROOT/${log_file}" >&3)
exec 2> >(tee -a "$LOGROOT/${err_file}" >&4)

echo "[INFO] stdout redirected $LOGROOT/${log_file}"
echo "[INFO] stderr redirected $LOGROOT/${err_file}"

echo "========== SLURM ENVIRONMENT DUMP =========="
env | grep '^SLURM' | sort
echo "============================================"

set -euo pipefail

################################################################################
HEAD_NODE="$(echo "$SLURM_NODELIST" | sed 's/[][]//g' | cut -d, -f1)"
export HEAD_NODE

export RAY_PORT="${RAY_PORT:-6379}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
echo "[INFO] RAY_PORT=$RAY_PORT"
echo "[INFO] DASHBOARD_PORT=$DASHBOARD_PORT"

################################################################################
#export SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
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
source "${SCRIPT_DIR}/models/qwen3-30B-A3B-union.sh"
export WANDB_KEY="$(cat ${wandb_key_file})"
export WANDB_GROUP=${model_name}-${run_postfix}

# Async training requires decoupled rollout resources.
# Allow overriding via environment variables so the split can match the allocated GPUs.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"

if ! [[ "$ROLLOUT_NUM_GPUS" =~ ^[0-9]+$ ]] || ! [[ "$ROLLOUT_NUM_GPUS_PER_ENGINE" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS}) and ROLLOUT_NUM_GPUS_PER_ENGINE (${ROLLOUT_NUM_GPUS_PER_ENGINE}) must be integers." >&2
  exit 1
fi

if (( ROLLOUT_NUM_GPUS_PER_ENGINE <= 0 )); then
  echo "[ERROR] ROLLOUT_NUM_GPUS_PER_ENGINE must be greater than 0." >&2
  exit 1
fi

if (( ROLLOUT_NUM_GPUS < ROLLOUT_NUM_GPUS_PER_ENGINE )); then
  echo "[ERROR] ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS}) must be >= ROLLOUT_NUM_GPUS_PER_ENGINE (${ROLLOUT_NUM_GPUS_PER_ENGINE})." >&2
  exit 1
fi

if (( ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
  echo "[ERROR] ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS}) must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE (${ROLLOUT_NUM_GPUS_PER_ENGINE})." >&2
  exit 1
fi

if ! [[ "$ACTOR_NUM_NODES" =~ ^[0-9]+$ ]] || (( ACTOR_NUM_NODES <= 0 )); then
  echo "[ERROR] ACTOR_NUM_NODES (${ACTOR_NUM_NODES}) must be a positive integer." >&2
  exit 1
fi

if ! [[ "$ACTOR_GPUS_PER_NODE" =~ ^[0-9]+$ ]] || (( ACTOR_GPUS_PER_NODE <= 0 )); then
  echo "[ERROR] ACTOR_GPUS_PER_NODE (${ACTOR_GPUS_PER_NODE}) must be a positive integer." >&2
  exit 1
fi

echo "[INFO] Async resource split: actor=${ACTOR_NUM_NODES}x${ACTOR_GPUS_PER_NODE} GPUs, rollout=${ROLLOUT_NUM_GPUS} GPUs (per engine ${ROLLOUT_NUM_GPUS_PER_ENGINE})."
echo "[INFO] Ensure the Slurm allocation covers actor + rollout GPU pools (see docs/en/examples/qwen3-4B.md)."
if command -v nvidia-smi >/dev/null 2>&1; then
  local_gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | xargs)
  if [[ "$local_gpu_count" =~ ^[0-9]+$ ]]; then
    total_allocated_guess=$(( SLURM_NNODES * local_gpu_count ))
    total_requested=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_NUM_GPUS ))
    if (( total_requested > total_allocated_guess )); then
      echo "[WARN] Requested actor+rollout GPUs (${total_requested}) exceed naive capacity estimate (${total_allocated_guess}). Update ACTOR_* or ROLLOUT_* or request more GPUs." >&2
    fi
  fi
fi

# model args
CKPT_ARGS=(
  --hf-checkpoint ${hf_ckpt_path}
  --ref-load ${dist_ckpt_path}
  --load ${megatron_ckpt_path}
  --save ${megatron_ckpt_path}
  --save-interval 25
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
  --rollout-batch-size 128
  --n-samples-per-prompt 8
  --rollout-max-response-len 8192
  --rollout-temperature 0.8
  --num-steps-per-rollout 4
  --global-batch-size 256
  --balance-data
  --seed 2333         # 🔎
  --rollout-seed 2333 # 🔎
)

EVAL_ARGS=(
  --eval-interval 25
  --eval-prompt-data ${eval_prompt_data}
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 0.7
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 8
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
  --kl-loss-coef 0.001
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

# Build the runtime environment JSON with proper variable substitution
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
    -- python3 train_async.py \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
      --reverse-routing-replay-order \
      --routing-replay-union \
      --keep-old-actor \
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
