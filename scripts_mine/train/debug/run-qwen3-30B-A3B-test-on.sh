#!/bin/bash

TIME=$(date +%Y%m%d-%H%M%S)
LOGROOT="/mnt/weka/home/haolong.jia/workspace/rlhf/slime/logs/sbatch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$LOGROOT"

echo "========== SLURM ENVIRONMENT DUMP =========="
env | grep '^SLURM' | sort
echo "============================================"

set -ex

# for rerun the task
cd /root/slime
pip install -e .
cp -rf /root/out-Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py /root/Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py
cp -rf /root/out-Megatron-LM/megatron/training/checkpointing.py /root/Megatron-LM/megatron/training/checkpointing.py
cp -rf /root/out-Megatron-LM/megatron/core/transformer/moe/moe_utils.py /root/Megatron-LM/megatron/core/transformer/moe/moe_utils.py
#cp -rf /root/out-Megatron-LM/megatron/training/arguments.py /root/Megatron-LM/megatron/training/arguments.py
#cp -rf /root/out-Megatron-LM/megatron/training/global_vars.py /root/Megatron-LM/megatron/training/global_vars.py

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
pkill -9 redis 2>/dev/null || true

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

################################################################################
HEAD_NODE=$(echo "$SLURM_NODELIST" | sed 's/[][]//g;s/,.*//')
export MASTER_ADDR="$HEAD_NODE"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR (HEAD_NODE)"

export RAY_PORT="${RAY_PORT:-6379}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
echo "[INFO] RAY_PORT=$RAY_PORT"
echo "[INFO] DASHBOARD_PORT=$DASHBOARD_PORT"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(ip -o -4 route get 1.1.1.1 | awk '{print $5}')}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[INFO] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "[INFO] NCCL_DEBUG=$NCCL_DEBUG"

################################################################################
#export SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SCRIPT_DIR=/root/slime/scripts_mine

export model_name="Qwen3-30B-A3B"
#export  run_postfix="$TIME"
export run_postfix="test-on"
export wandb_key_file="scripts_mine/wandb_key_sandy.txt"

export hf_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint/${model_name}"
export dist_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint_torch_dist/${model_name}"
export megatron_ckpt_path="/mnt/sharefs/users/haolong.jia/checkpoint_megatron/${model_name}-${run_postfix}"
export start_rollout_id=$(
  val=$(cat "${megatron_ckpt_path}/latest_checkpointed_iteration.txt" 2>/dev/null || true)
  if [[ "$val" =~ ^[0-9]+$ ]]; then echo $((val + 1)); else echo ""; fi
)

export prompt_data="/mnt/sharefs/users/haolong.jia/RL-data/dapo-math-17k/dapo-math-17k.jsonl"
export eval_prompt_data="aime /mnt/sharefs/users/haolong.jia/RL-data/aime-2024/aime-2024.jsonl"
################################################################################

source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"
export WANDB_KEY=$(cat ${wandb_key_file})
export WANDB_GROUP=${model_name}-${run_postfix}

CKPT_ARGS=(
  --hf-checkpoint ${hf_ckpt_path}
  --ref-load ${dist_ckpt_path}
  --load ${megatron_ckpt_path}
  --save ${megatron_ckpt_path}
  --save-interval 50
  #  --no-save-optim
  #  --no-save-rng
)

if [ -n "$start_rollout_id" ]; then
  CKPT_ARGS+=(--start-rollout-id "$start_rollout_id") # ------ to continue from previous run ------
fi

ROLLOUT_ARGS=(
  --prompt-data ${prompt_data}
  --input-key prompt
  --label-key label
  --apply-chat-template
  # --rollout-shuffle  # 注释掉以保持数据顺序一致
  --rm-type deepscaler
  --num-rollout 10
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --rollout-max-response-len 8192
  --rollout-temperature 0.8
  --num-steps-per-rollout 1

  --global-batch-size 128
  --balance-data
  --seed 2333         # 🔎
  --rollout-seed 2333 # 🔎
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

  # --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
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
  --sglang-mem-fraction-static 0.7
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
  --sglang-enable-deterministic-inference # 🔎
  --sglang-attention-backend flashinfer   # 🔎
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
  --deterministic-mode # 🔎
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\"
  }
}"

# launch the master node of ray in container
if [ "$(hostname -s)" = "$HEAD_NODE" ]; then
  ray start --head \
    --node-ip-address "$MASTER_ADDR" \
    --port "$RAY_PORT" \
    --num-gpus 8 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port "$DASHBOARD_PORT" \
    # --disable-usage-stats
else
  ray start --address "$MASTER_ADDR:$RAY_PORT" \
    --num-gpus 8 \
    --disable-usage-stats
fi

if [ "$(hostname -s)" = "$HEAD_NODE" ]; then
  #--debug-train-only \
  ray job submit --address="http://${MASTER_ADDR}:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes "${SLURM_NNODES}" \
    --actor-num-gpus-per-node 8 \
    --colocate \
    --use-routing-replay \
    --save-debug-rollout-data /mnt/sharefs/users/haolong.jia/RL-data/DEBUG/on/dapo-math_{rollout_id}.pt \
    --save-debug-train-output /mnt/sharefs/users/haolong.jia/RL-output/DEBUG/on/dapo-math_output_{rollout_id}_{rank}.pt \
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
else
  sleep inifnity
fi
