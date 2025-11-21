import os
import torch


ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self, use_pre_softmax=False):
        self.use_pre_softmax = use_pre_softmax
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        # Store union masks if needed
        self.union_mask_list = []
        # Store scores for debugging/analysis
        self.scores_list = []
        self.num_experts_list = []
        # Pre-union cached recordings
        self.preunion_current_top_indices_list = []
        self.preunion_current_scores_list = []
        self.preunion_old_top_indices_list = []
        self.preunion_old_scores_list = []
        self.preunion_current_num_experts_list = []
        self.preunion_old_num_experts_list = []
        RoutingReplay.all_routing_replays.append(self)

    @torch.no_grad()
    def record(self, top_indices, scores=None, num_experts=None):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)
        self.union_mask_list.append(None) # Placeholder for union masks

        if scores is not None:
            buf = torch.empty_like(scores, device="cpu", pin_memory=True)
            buf.copy_(scores)
            self.scores_list.append(buf)
        else:
            self.scores_list.append(None)

        self.num_experts_list.append(num_experts)

    @torch.no_grad()
    def update(self, top_indices, index, union_mask_list=None, scores=None, num_experts=None):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list[index] = buf
        if union_mask_list is not None:
            buf = torch.empty_like(union_mask_list, device="cpu", pin_memory=True)
            buf.copy_(union_mask_list)
            self.union_mask_list[index] = buf
        if scores is not None:
            buf = torch.empty_like(scores, device="cpu", pin_memory=True)
            buf.copy_(scores)
            self.scores_list[index] = buf
        self.num_experts_list[index] = num_experts

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        union_mask = self.union_mask_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device()), union_mask.to(torch.cuda.current_device()) if union_mask is not None else None

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        union_mask = self.union_mask_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device()), union_mask.to(torch.cuda.current_device()) if union_mask is not None else None

    def _reset_storage(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.union_mask_list = []
        self.scores_list = []
        self.num_experts_list = []

    def clear(self):
        """Clear all recorded data and reset indices."""
        self._reset_storage()
        self.preunion_current_top_indices_list = []
        self.preunion_current_scores_list = []
        self.preunion_current_num_experts_list = []
        self.preunion_old_top_indices_list = []
        self.preunion_old_scores_list = []
        self.preunion_old_num_experts_list = []

    def reset_indices(self):
        """Reset forward/backward indices for replay without clearing recorded data.

        This is useful when you need to replay the same routing decisions multiple times.
        """
        self.forward_index = 0
        self.backward_index = 0

    def get_recorded_routing(self):
        """Get all recorded routing decisions for debugging/analysis.

        Returns:
            List of routing decision tensors (cloned to avoid modification).
        """
        return [idx.clone() for idx in self.top_indices_list]

    def get_recorded_scores(self):
        """Get all recorded expert scores for debugging/analysis.

        Returns:
            List of score tensors (cloned to avoid modification).
            Empty list if scores were not recorded.
        """
        return [score.clone() if score is not None else None for score in self.scores_list]

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()

    @torch.no_grad()
    def stash_preunion_records(self, target: str) -> None:
        """Move the current recording buffers into the requested pre-union cache."""
        if not self.top_indices_list:
            return

        if target == "current":
            self.preunion_current_top_indices_list = self.top_indices_list
            self.preunion_current_scores_list = self.scores_list
            self.preunion_current_num_experts_list = self.num_experts_list
        elif target == "old":
            self.preunion_old_top_indices_list = self.top_indices_list
            self.preunion_old_scores_list = self.scores_list
            self.preunion_old_num_experts_list = self.num_experts_list
        else:
            raise ValueError(f"Unknown pre-union target: {target}")

        # Reset primary buffers so the next record stage starts from scratch.
        self._reset_storage()

    def has_preunion_records(self) -> bool:
        return bool(self.preunion_current_top_indices_list) and bool(self.preunion_old_top_indices_list)

    @torch.no_grad()
    def build_preunion_union(self) -> None:
        """Combine recorded current/old routings before any forward replay."""
        if not self.has_preunion_records():
            raise RuntimeError("Pre-union routing replay requires both current and old recordings.")

        if len(self.preunion_current_top_indices_list) != len(self.preunion_old_top_indices_list):
            raise RuntimeError(
                "Mismatched number of routing records between current and old models for pre-union replay."
            )

        new_top_indices_list = []
        new_union_mask_list = []
        for idx, (current_idx, old_idx, num_experts) in enumerate(
            zip(
                self.preunion_current_top_indices_list,
                self.preunion_old_top_indices_list,
                self.preunion_current_num_experts_list,
            )
        ):
            if current_idx.shape != old_idx.shape:
                raise RuntimeError("Routing replay recordings must share the same shape per layer.")

            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = current_idx.device

            current_idx_dev = current_idx.to(device, non_blocking=True)
            old_idx_dev = old_idx.to(device, non_blocking=True)

            num_tokens, _ = current_idx.shape
            scores_shape = (num_tokens, num_experts)

            _, union_indices, selected_union_mask = combine_topk_indices(
                current_idx_dev,
                old_idx_dev,
                scores=None,
                scores_shape=scores_shape,
                scores_device=device,
                use_pre_softmax=self.use_pre_softmax,
            )

            # Logging for debugging
            topk = current_idx.shape[-1]
            if union_indices.shape[-1] != topk:
                import torch.distributed as dist
                if idx == 0:  # Only log once (after first pop)
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    if self.use_pre_softmax:
                        # selected_union_mask is float 1.0/0.0
                        mean_select = selected_union_mask.sum(dim=1).float().mean().item()
                    else:
                        # selected_union_mask is float 0.0/-inf
                        mean_select = (selected_union_mask != float('-inf')).sum(dim=1).float().mean().item()
                    if rank == 0:
                        print(f"[RoutingReplay Union] topk {topk} → {union_indices.shape[-1]} (max), {format(mean_select, '.2f')} (mean) experts per token")

            buf_indices = torch.empty_like(union_indices, device="cpu", pin_memory=True)
            buf_indices.copy_(union_indices)
            buf_union_mask = torch.empty_like(selected_union_mask, device="cpu", pin_memory=True)
            buf_union_mask.copy_(selected_union_mask)

            new_top_indices_list.append(buf_indices)
            new_union_mask_list.append(buf_union_mask)

        # only update the primary buffers, scores & num_experts are not needed for pre-union replay
        self.top_indices_list = new_top_indices_list
        self.union_mask_list = new_union_mask_list
        self.forward_index = 0
        self.backward_index = 0

        # Free cached recordings
        self.preunion_current_top_indices_list = []
        self.preunion_current_scores_list = []
        self.preunion_current_num_experts_list = []
        self.preunion_old_top_indices_list = []
        self.preunion_old_scores_list = []
        self.preunion_old_num_experts_list = []


def combine_topk_indices(
    top_indices,
    current_top_indices,
    scores=None,
    use_pre_softmax=False,
    *,
    scores_shape=None,
    scores_device=None,
):
    """Combine recorded and current top-k indices with union semantics.

    This function computes the union of recorded and current expert selections,
    then ensures all tokens select exactly ``max_select`` experts (the maximum union size).

    Args:
        top_indices: Previously recorded expert indices ``[num_tokens, topk]``.
        current_top_indices: Current expert indices ``[num_tokens, topk]``.
        scores: Router scores ``[num_tokens, num_experts]``. Optional.
            If ``None``, the caller must provide ``scores_shape`` and ``scores_device`` so
            synthetic scores can be created for padding/randomization. In that case the
            returned ``probs`` will be ``None``.
        use_pre_softmax: If ``True``, return probs & selected union mask for pre-softmax masking.
        scores_shape: Shape tuple ``(num_tokens, num_experts)`` to use when ``scores`` is ``None``.
        scores_device: Torch device for the synthetic scores / union mask when ``scores`` is ``None``.

    Returns:
        probs: Scores for selected experts ``[num_tokens, max_select]`` or ``None`` if ``scores`` is ``None``.
        top_indices: Selected expert indices ``[num_tokens, max_select]``.
        selected_union_mask: Mask indicating which experts are in union ``[num_tokens, max_select]``.
    """
    if scores is None:
        if scores_shape is None or scores_device is None:
            raise ValueError("scores_shape and scores_device must be provided when scores is None.")
        base_scores = torch.rand(scores_shape, dtype=torch.float32, device=scores_device)
        return_probs = False
    else:
        base_scores = scores
        return_probs = True
        if scores_shape is None:
            scores_shape = scores.shape
        if scores_device is None:
            scores_device = scores.device

    # Step 1: Compute union of recorded and current expert selections
    union_indices = torch.cat([current_top_indices, top_indices], dim=-1)
    union_mask = torch.zeros(scores_shape, dtype=torch.bool, device=scores_device)
    union_mask.scatter_(dim=-1, index=union_indices, value=True)

    num_selects = union_mask.sum(-1)  # [num_tokens], number of experts in union per token
    max_select = num_selects.max().item()  # Maximum union size across all tokens

    # Step 2: Ensure all tokens select exactly max_select experts
    # Use score boosting to guarantee all union experts are selected first,
    # then pad with highest-scoring non-union experts for tokens with union_size < max_select
    masked_scores = base_scores.clone()
    boost_value = base_scores.abs().max() + 100.0  # Large enough to ensure union experts rank highest
    masked_scores = torch.where(union_mask, masked_scores + boost_value, masked_scores)

    # Select top max_select experts (includes all union experts + padding)
    _, top_indices = torch.topk(masked_scores, k=max_select, dim=1, sorted=False)

    # Step 3: Gather probs & selected_union_mask if needed
    if use_pre_softmax:  # probs already softmaxed before, mask should be 1/0
        # selected_union_mask: 1.0 for experts that are in the union, 0.0 for padding
        selected_union_mask = union_mask.gather(1, top_indices).float()  # [num_tokens, max_select]
        # adjust scores for the selected experts
        if return_probs:
            probs = base_scores.gather(1, top_indices)
            probs = probs * selected_union_mask  # Zero out padding experts
        else:
            probs = None

    else:  # probs will be softmaxed later, mask should be 0/-inf
        # selected_union_mask: 0.0 for experts that are in the union, -inf for padding
        selected_union_mask = torch.where(
            union_mask.gather(1, top_indices),
            torch.zeros_like(top_indices, dtype=torch.float32, device=scores_device),
            torch.full_like(top_indices, float('-inf'), dtype=torch.float32, device=scores_device)
        )  # [num_tokens, max_select]
        # adjust scores for the selected experts
        if return_probs:
            probs = base_scores.gather(1, top_indices)
            probs = probs + selected_union_mask  # Set padding experts to -inf
        else:
            probs = None

    # Sanity check: detect NaN/Inf early (only in debug mode)
    # if torch.isnan(probs).any() or torch.isinf(probs).any():
    #     import torch.distributed as dist
    #     rank = dist.get_rank() if dist.is_initialized() else 0
    #     print(f"[WARNING] Rank {rank}: Detected NaN/Inf in routing probs after combine_topk_indices!")
    #     print(f"  probs stats: min={probs.min()}, max={probs.max()}, nan={torch.isnan(probs).sum()}, inf={torch.isinf(probs).sum()}")

    return probs, top_indices, selected_union_mask


def get_routing_replay_compute_topk(old_compute_topk, use_pre_softmax=False):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]

            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            elif routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                # Record both routing decisions and scores for debugging
                ROUTING_REPLAY.record(
                    top_indices,
                    scores=scores if os.environ.get("ROUTING_REPLAY_RECORD_SCORES", "0") == "1" else None,
                    num_experts=scores.shape[-1]
                )
                # Logging for debugging
                if len(ROUTING_REPLAY.top_indices_list) == 1:  # Only log once (after first record)
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    if rank == 0:
                        print(f"[RoutingReplay Record] first token indices:\n{top_indices[:1, :]}")

            elif routing_replay_stage == "union":
                # In union mode, we calculate the topk indices as usual,
                # However, we will combine them with recorded indices for this forward, and update the recorded indices.
                # This can be treated as "replay_forward" + "record" in one pass.
                recorded_top_indices, _ = ROUTING_REPLAY.pop_forward()  # Get recorded indices (union_mask is None in record stage)
                _, current_top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                probs, top_indices, selected_union_mask = combine_topk_indices(recorded_top_indices, current_top_indices, scores, use_pre_softmax=use_pre_softmax)
                # Logging for debugging
                if top_indices.shape[-1] != topk:
                    import torch.distributed as dist
                    if ROUTING_REPLAY.forward_index == 1:  # Only log once (after first pop)
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        if use_pre_softmax:
                            # selected_union_mask is float 1.0/0.0
                            mean_select = selected_union_mask.sum(dim=1).float().mean().item()
                        else:
                            # selected_union_mask is float 0.0/-inf
                            mean_select = (selected_union_mask != float('-inf')).sum(dim=1).float().mean().item()
                        if rank == 0:
                            print(f"[RoutingReplay Union] topk {topk} → {top_indices.shape[-1]} (max), {format(mean_select, '.2f')} (mean) experts per token")
                            print(f"[RoutingReplay Union] first token probs:\n{probs[:1, :]}")

                # Update recorded indices with union results
                ROUTING_REPLAY.update(
                    top_indices,
                    ROUTING_REPLAY.forward_index - 1,  # index should -1 as we have updated forward_index
                    selected_union_mask,
                    scores=scores if os.environ.get("ROUTING_REPLAY_RECORD_SCORES", "0") == "1" else None,
                    num_experts=scores.shape[-1]
                )

            elif routing_replay_stage == "replay_forward":
                top_indices, selected_union_mask = ROUTING_REPLAY.pop_forward()
                if selected_union_mask is not None:
                    # If selected_union_mask is provided, we need to mask the probs accordingly
                    # And the top_indices may have more than topk experts per token
                    if use_pre_softmax:
                        # For pre-softmax masking, selected_union_mask is float 1.0/0.0
                        probs = scores.gather(1, top_indices) * selected_union_mask
                    else:
                        # For post-softmax masking, selected_union_mask is float 0.0/-inf
                        probs = scores.gather(1, top_indices) + selected_union_mask
                else:
                    assert (
                        top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                    ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                    probs = scores.gather(1, top_indices)
                # Logging for debugging
                if ROUTING_REPLAY.forward_index == 1 and selected_union_mask is not None:  # Only log once (after first forward)
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    if rank == 0:
                        if use_pre_softmax:
                            topk = selected_union_mask.sum(dim=1).int().tolist()[:16]
                        else:
                            topk = (selected_union_mask != float('-inf')).sum(dim=1).int().tolist()[:16]
                        print(f"[RoutingReplay Forward] first 16 tokens selected experts counts: {topk}")
                        print(f"[RoutingReplay Forward] first token indices:\n{top_indices[:1, :]}")
                        print(f"[RoutingReplay Forward] first token probs:\n{probs[:1, :]}")

            elif routing_replay_stage == "replay_backward":
                top_indices, selected_union_mask = ROUTING_REPLAY.pop_backward()
                if selected_union_mask is not None:
                    # If selected_union_mask is provided, we need to mask the probs accordingly
                    # And the top_indices may have more than topk experts per token
                    if use_pre_softmax:
                        # For pre-softmax masking, selected_union_mask is float 1.0/0.0
                        probs = scores.gather(1, top_indices) * selected_union_mask
                    else:
                        # For post-softmax masking, selected_union_mask is float 0.0/-inf
                        probs = scores.gather(1, top_indices) + selected_union_mask
                else:
                    assert (
                        top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                    ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                    probs = scores.gather(1, top_indices)

            else:
                raise ValueError(f"Unknown ROUTING_REPLAY_STAGE: {routing_replay_stage}")

            return probs, top_indices

        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module, use_pre_softmax=False):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay(use_pre_softmax=use_pre_softmax)

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
        print(f"[RoutingReplay] Registered routing replay for module {module}")
