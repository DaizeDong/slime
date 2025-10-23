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

5.  `--save-debug-train-output /your/saved/debug/output_{rollout_id}.pt`

    When enabled, per-batch model outputs during training will be saved for comprehensive debugging. The file is saved using the format: `args.save_debug_train_output.format(rollout_id=rollout_id, rank=rank)`.
    
    **Saved data format**: The saved file contains a dictionary with the following structure:
    ```python
    {
        "rollout_id": int,           # Rollout identifier
        "rank": int,                 # GPU rank
        "outputs": [                 # List of per-batch outputs
            {
                "tokens": list[torch.Tensor],              # Token IDs for each sample
                "old_log_probs": list[torch.Tensor],       # Old policy log probabilities (per-token)
                "curr_log_probs": list[torch.Tensor],      # Current policy log probabilities (per-token)
                "importance": list[torch.Tensor],          # Per-token importance = exp(curr - old)
                "entropy": list[torch.Tensor],             # Per-token entropy
                "advantages": list[torch.Tensor],          # Per-token advantages
                "response_lengths": list[int],             # Response length for each sample
                "total_lengths": list[int],                # Total length for each sample
                "metrics": {                               # Aggregated metrics
                    "pg_loss": torch.Tensor,               # Policy gradient loss
                    "entropy_loss": torch.Tensor,          # Entropy loss
                    "pg_clipfrac": torch.Tensor,           # PPO clipping fraction
                    "ppo_kl": torch.Tensor,                # PPO KL divergence
                    # Optional fields (if enabled):
                    "kl_loss": torch.Tensor,               # KL loss (if args.use_kl_loss)
                    "tis": torch.Tensor,                   # TIS weight (if args.use_tis)
                    "ois": torch.Tensor,                   # OIS weight (if args.use_tis)
                    "tis_clipfrac": torch.Tensor,          # TIS clipping fraction (if args.use_tis)
                },
                # Optional TIS fields (if args.use_tis):
                "tis_concat": torch.Tensor,                # Concatenated TIS weights (all tokens)
                "ois_concat": torch.Tensor,                # Concatenated OIS weights (all tokens)
            },
            # ... more batches
        ]
    }
    ```
    
    This comprehensive output allows for detailed analysis of the training process, including tracking per-token importance weights (which represent the ratio between current and old policies), and all loss components.