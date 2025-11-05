# Debugging

## Aligning Precision

During the development of slime, it is often necessary to check if the model's precision is correct. This can be verified in the following ways:

1.  **First Training Step**
    1.  Check if the generated `rollout` is coherent. If not, there are two possible reasons:
        * Parameters were not loaded correctly. You need to check the logs for a confirmation that Megatron successfully loaded the checkpoint (ckpt).
        * There was an error in updating the parameters. You can check if all parameters were converted and mapped correctly, or if the parameter names were converted according to the parallelization strategy (e.g., when `pp_size > 1`, check if the layer IDs for the parameters provided by the second stage are correct). A thorough method is to save all parameters in the `load_weights` implementation of the corresponding model in SGLang and verify that they are consistent with the loaded checkpoint.
        * If all parameters are updated correctly and the problem persists, it's possible that some special buffers in SGLang were released during the release process.
        * If you are testing with a pretrained model, you can switch to an instruct version of a model with the same architecture to see if this garbled output is specific to the pretrained model.

    2.  Check the printed rollout stats to see if `log_probs` and `ref_log_probs` are exactly equal (meaning KL divergence is 0 in the first step) and their values are small.
        * If they are not exactly equal, it is usually caused by certain non-deterministic kernels in the Transformer Engine, for example:
            * In some versions of Transformer Engine (TE), Megatron requires `--attention-backend flash` to enforce the use of Flash Attention, thereby avoiding numerical instability from the fused attention under Context Parallelism (CP).
        * If the values are large (e.g., > 1), there are generally two possibilities:
            * If the value is extremely large, there is likely a problem with the training configuration.
            * If the value is only slightly larger than the SFT loss, for example, if the log probability of an instruct model reaches 0.8, it might be because the data does not conform to the trained chat template or does not match the cold-start distribution.

    3.  When running one inference step per training step (`num_steps_per_rollout == 1`), check if the KL divergence is 0 and if the `grad_norm` is small.
        * This is basically due to some Megatron / TE related bugs, for example:
            * Mixture of Experts (MoE) requires enabling `--moe-permute-fusion`.

2.  **Second Training Step**
    1.  For integrated training and inference, check if the second step can be loaded correctly and whether it results in an Out of Memory (OOM) error.

## Separate Debugging for Training and Inference

slime supports debugging the training and inference parts separately, which allows for the following:

* When tuning/debugging the inference part, you can start the task with only a few GPUs.
* When tuning/debugging the training part, you can ensure the model input is fixed, removing the randomness of rollouts.

Specifically, slime currently provides the following parameters for separate debugging:

1.  `--debug-rollout-only`

    When enabled, slime will not load Megatron and will only initialize SGLang. You can use this method to debug the inference part.

2.  `--debug-train-only`

    When enabled, slime will not load SGLang and will only initialize Megatron. You can use this method to debug the training part.

3.  `--save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

    When enabled, the results of each rollout will be saved. This can be used in conjunction with `--debug-rollout-only`. Note that the data is saved using the format: `args.save_debug_rollout_data.format(rollout_id=rollout_id)`.

4.  `--load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

    When enabled, data will be loaded from `args.load_debug_rollout_data.format(rollout_id=rollout_id)`, and SGLang will not be initialized (automatically setting `debug_train_only=True`). This method allows you to fix the input for the training part to tune it, for example, by switching between different parallelization strategies.

5.  `--save-debug-train-output /your/saved/debug/output_{rollout_id}_{rank}.pt`

    When enabled, detailed training outputs will be saved for each training iteration to analyze the training process. The save path format is `args.save_debug_train_output.format(rollout_id=rollout_id, rank=rank)`.
    
    The saved data includes:
    - `rollout_id`: Current rollout ID
    - `rank`: Current process rank
    - `role`: Model role (actor/critic)
    - `num_steps`: Number of training steps
    - `steps`: Detailed information for each training step, including:
      - `step_id`: Step ID
      - `loss_dict`: Loss dictionary (containing various loss metrics)
      - `grad_norm`: Gradient norm
      - `debug_data`: Detailed token-level data, including:
        - `unconcat_tokens`: Token sequences for each sample
        - `response_lengths`: Response length for each sample
        - `total_lengths`: Total length for each sample
        - `loss_masks`: Loss mask for each sample
        - `advantages`: Advantage values (if available)
        - `returns`: Return values (if available)
        - `old_log_probs`: Old policy log probabilities (if available)
        - `ref_log_probs`: Reference model log probabilities (if available)
        - `values`: Value function outputs (if available)
        - `current_log_probs`: Current policy log probabilities
        - `current_entropy`: Current policy entropy (if available)
        - **`policy_importance_ratio`**: **PPO/GRPO importance weights** $\frac{\pi_\theta}{\pi_{old}} = \exp(\log \pi_\theta - \log \pi_{old})$ (per token)
        - `tis_importance_weights`: TIS importance weights (if TIS is enabled, used for trajectory importance sampling)
        - `logits_sample`: Sample of logits (for inspection)
    
    **Saving Strategy**:
    - **All DP ranks will save**: Because different DP ranks process different data batches, all data needs to be saved
    - **Only TP rank 0 saves**: Because log_probs and other results are synchronized across TP ranks via all_reduce, all TP ranks have identical values
    - **Only the last PP stage saves**: Because only the last pipeline stage has complete logits output
    - The saved data includes a `parallel_info` field that records the DP/TP/PP/CP rank information for this file
    
    **TP Synchronization Note**: All `log_probs` and `entropy` values are already synchronized across TP ranks via all_reduce operations inside the `calculate_log_probs_and_entropy` function, ensuring that all TP ranks have consistent values.
    
    **Memory Optimization Note** (Critical!):
    - All collected tensors use `.detach()` to **break the computation graph**, preventing gradient information from being retained
    - Then use `.to("cpu", non_blocking=True)` for **asynchronous CPU transfer**, immediately freeing GPU memory
    - **Most Critical**: debug_data is **NOT returned through Megatron's logging_dict**! Megatron accumulates all microbatches' logging_dict in GPU memory, causing OOM. We use a separate global buffer to avoid GPU accumulation

6.  `--save-debug-train-output` **Additional Feature: Record vs Train Comparison**

    When `--save-debug-train-output` is enabled along with `--use-routing-replay`, an additional record stage (first forward pass) data file is saved:
    
    **Saved Files**:
    - `output_{rollout_id}_{rank}.pt` - Train stage data (second forward + backward pass)
    - `output_{rollout_id}_{rank}_record.pt` - Record stage data (first forward pass for computing log_probs)
    
    **Record File Contains**:
    - `log_probs`: Log probabilities computed from the first forward pass
    - `routing_decisions`: MoE layer routing decisions (if routing_replay is enabled)
    - `routing_scores`: MoE layer expert scores (if routing_replay is enabled) - **New**
    - `response_lengths`, `total_lengths`: Sequence length information
    - `parallel_info`: Parallel configuration information
    
    **Use Comparison Script to Verify Routing Replay**:
    ```bash
    python examples/debug_analysis/compare_record_train.py \
        --rollout-id 0 \
        --rank 0 \
        --debug-dir /path/to/DEBUG/
    ```
    
    The script will:
    - ✅ Compare log_probs from both forward passes (should be identical)
    - ✅ Compare MoE routing decisions (should be 100% consistent)
    - ✅ Compare MoE expert scores numerical changes - **New**
    - ✅ Analyze any inconsistencies and their extent
    
    **About routing_scores**:
    - Scores record each token's affinity to all experts
    - These scores are **currently used only for debugging/analysis**, not for replay forward
    - May be used in future for advanced routing strategies (e.g., adaptive routing based on historical scores)
    
    **Why Two Forward Passes?**
    - **First (record)**: Compute "old policy" log_probs, record MoE routing decisions, **no parameter updates**
    - **Second (train)**: Actual training forward+backward pass, uses recorded routing decisions, **updates parameters**
    - Although both use the same parameters, they serve completely different purposes: first is data collection, second is learning