# Debug 指南

## 对齐精度

在开发 slime 的过程中，经常会需要检查模型的精度是否正确，可以通过以下方式检查：

1. 训练第一步
   1. rollout 的生成是否是人话，如果不是，有以下 2 种可能：
      - 参数没有正常加载。需要查看是否有 megatron 成功加载 ckpt 的日志；
      - 更新参数有误。可以查看是不是所有的参数都做了转换和参数对应，或者参数名是不是根据并行做了转换（例如 pp_size > 1 时，第二个 stage 提供的参数的 layer id 是不是正确的）。一个比较彻底的方法是在对应模型的 sglang 实现的 `load_weights` 中保存所有的参数，查看和加载的 ckpt 中是否一致；
      - 如果所有参数更新都正确，还出现问题，有可能是 sglang 里有一些特殊的 buffer 在 release 的时候被释放了；
      - 如果是用 pretrain 模型进行的测试，可以换成同结构模型的 instruct 版本，查看这种乱码是不是 pretrain 模型特有的。
   2. 查看打印的 rollout stats 的 `log_probs` 和 `ref_log_probs` 是否完全相等（即第一步 kl=0），且值较小
      - 如果不是完全相等的，一般是 transformer engine 中的某些 non-deterministic kernel 导致的，例如：
        - 在某些版本的 te 里，megatron 需要 `--attention-backend flash`，来强制使用 flash attention，从而避免 CP 下 fused attention 的数值不稳定；
      - 如果数值较大（例如 >1），一般有 2 种可能：
        - 如果值非常大，应该是训练配置有问题；
        - 如果值只是比 sft loss 的状态略大，例如 instruct 模型的 logprob 到了 0.8，有可能是数据不符合训练的 chat template，或者不符合冷启动的分布。
   3. 查看在推一训一（`num_steps_per_rollout == 1`），kl 是否为 0，grad_norm 是否较小
      - 基本上就是一些 megatron / te 相关的 bug，例如：
        - moe 需要开启 `--moe-permute-fusion`。

2. 训练第二步
   1. 对于训推一体，查看是否能正确加载第二步，是否会 OOM；

## 训练推理单独 debug

slime 支持将训练部分和推理部分分开进行调试，从而实现：

- 在调优/debug 推理部分时，只用少量卡就可以启动任务；
- 在调优/debug 训练部分时，可以保证模型输入固定，去除 rollout 的随机性。

具体来说，目前 slime 提供了如下的参数来进行分离调试：

1. `--debug-rollout-only`

   开启后，slime 将不会加载 megatron，只初始化 sglang ，可以用这个方法来进行推理部分的调试。

1. `--debug-train-only`

   开启后，slime 将不会加载 sglang，只初始化 megatron ，可以用这个方法来进行训练部分的调试。

2. `--save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

   开启后，会保存每次 rollout 的结果，可以和 `--debug-rollout-only` 配合使用。注意保存的方式为 `args.save_debug_rollout_data.format(rollout_id=rollout_id)`。

3. `--load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

   开启后，会从 `args.load_debug_rollout_data.format(rollout_id=rollout_id)` 来加载数据，并且不会初始化 sglang（自动设置 `debug_train_only=True`）。可以以这种方式来固定训练部分的输入，对训练部分进行调优，例如切换各种并行。

4. `--save-debug-train-output /your/saved/debug/output_{rollout_id}_{rank}.pt`

   开启后，会保存每次训练的详细输出，用于分析训练过程。保存路径格式为 `args.save_debug_train_output.format(rollout_id=rollout_id, rank=rank)`。
   
   保存的数据包括：
   - `rollout_id`: 当前的 rollout ID
   - `rank`: 当前进程的 rank
   - `role`: 模型角色（actor/critic）
   - `num_steps`: 训练步数
   - `steps`: 每个训练步的详细信息，包括：
     - `step_id`: 步骤 ID
     - `loss_dict`: 损失字典（包含各种损失指标）
     - `grad_norm`: 梯度范数
     - `debug_data`: 详细的 token-level 数据，包括：
       - `unconcat_tokens`: 每个样本的 token 序列
       - `response_lengths`: 每个样本的回复长度
       - `total_lengths`: 每个样本的总长度
       - `loss_masks`: 每个样本的损失掩码
       - `advantages`: 优势值（如果有）
       - `returns`: 回报值（如果有）
       - `old_log_probs`: 旧策略的对数概率（如果有）
       - `ref_log_probs`: 参考模型的对数概率（如果有）
       - `values`: 价值函数输出（如果有）
       - `current_log_probs`: 当前策略的对数概率
       - `current_entropy`: 当前策略的熵（如果有）
       - **`policy_importance_ratio`**: **PPO/GRPO 重要性权重** $\frac{\pi_\theta}{\pi_{old}} = \exp(\log \pi_\theta - \log \pi_{old})$ （每个 token）
       - `tis_importance_weights`: TIS 重要性权重（如果启用 TIS，用于轨迹重要性采样）
       - `logits_sample`: 部分 logits 样本（用于检查）
   
   **保存策略说明**：
   - **每个 DP rank 都会保存**：因为不同 DP rank 处理不同的数据batch，需要保存所有数据
   - **只在 TP rank 0 保存**：因为 TP 各个 rank 的 log_probs 等结果通过 all_reduce 同步，所有 TP rank 的结果相同
   - **只在最后一个 PP stage 保存**：因为只有最后的 pipeline stage 才有完整的 logits 输出
   - 保存的数据中包含 `parallel_info` 字段，记录了该文件对应的 DP/TP/PP/CP rank 信息
   
   **TP 同步说明**：所有的 `log_probs` 和 `entropy` 值都已经在 `calculate_log_probs_and_entropy` 函数内通过 all_reduce 在 TP group 间同步，因此所有 TP rank 的值是一致的。
   
   **显存优化说明**（关键！）：
   - 所有收集的 tensor 都使用 `.detach()` **断开计算图**，防止梯度信息被保留
   - 然后使用 `.to("cpu", non_blocking=True)` **异步传输到 CPU**，立即释放 GPU 显存
   - **最关键**：debug_data **不通过 Megatron 的 logging_dict 返回**！Megatron 会在 GPU 上累积所有 microbatch 的 logging_dict，导致 OOM。我们使用独立的全局 buffer 存储，避免在 GPU 上累积

5. `--save-debug-train-output` **附加功能：record vs train 对比**

   当启用 `--save-debug-train-output` 且使用 `--use-routing-replay` 时，会额外保存 record 阶段（第一次前向）的数据到单独的文件：
   
   **保存的文件**：
   - `output_{rollout_id}_{rank}.pt` - train 阶段数据（第二次前向 + 反向传播）
   - `output_{rollout_id}_{rank}_record.pt` - record 阶段数据（第一次前向，用于计算 log_probs）
   
   **record 文件包含**：
   - `log_probs`: 第一次前向计算的 log_probs
   - `routing_decisions`: MoE 层的 routing 决策（如果启用 routing_replay）
   - `routing_scores`: MoE 层的 expert 分数（如果启用 routing_replay）- **新增**
   - `response_lengths`, `total_lengths`: 序列长度信息
   - `parallel_info`: 并行配置信息
   
   **使用对比脚本验证 routing replay**：
   ```bash
   python examples/debug_analysis/compare_record_train.py \
       --rollout-id 0 \
       --rank 0 \
       --debug-dir /path/to/DEBUG/
   ```
   
   脚本会：
   - ✅ 对比两次前向的 log_probs 是否一致（应该完全相同）
   - ✅ 对比 MoE routing 决策是否完全重放（应该 100% 一致）
   - ✅ 对比 MoE expert scores 的数值变化 - **新增**
   - ✅ 分析任何不一致的来源和程度
   
   **关于 routing_scores**：
   - Scores 记录了每个 token 对所有 experts 的亲和度分数
   - 这些分数**目前仅用于调试分析**，不参与 replay forward
   - 未来可能用于实现更高级的 routing 策略（如基于历史 scores 的自适应 routing）
   
   **为什么需要两次前向？**
   - **第一次（record）**：计算"旧策略"的 log_probs，记录 MoE routing 决策，**不更新参数**
   - **第二次（train）**：真正的训练前向+反向传播，使用记录的 routing 决策，**更新参数**
   - 虽然两次都用相同的参数，但目的完全不同：第一次是收集数据，第二次是学习