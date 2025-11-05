import os
import torch


ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.union_mask_list = [] # Store union masks if needed
        self.scores_list = []  # Store scores for debugging/analysis
        RoutingReplay.all_routing_replays.append(self)

    @torch.no_grad()
    def record(self, top_indices, scores=None):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)
        self.union_mask_list.append(None) # Placeholder for union masks
        if scores is not None:
            buf = torch.empty_like(scores, device="cpu", pin_memory=True)
            buf.copy_(scores)
            self.scores_list.append(buf)

    @torch.no_grad()
    def update(self, top_indices, index, union_mask_list=None, scores=None):
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

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.union_mask_list = []
        self.scores_list = []

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
        return [score.clone() for score in self.scores_list]

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()


def combine_topk_indices(top_indices, current_top_indices, scores):
    """Combine recorded and current top-k indices with union semantics.
    
    This function computes the union of recorded and current expert selections,
    then ensures all tokens select exactly max_select experts (the maximum union size).
    
    CRITICAL: This increases the number of selected experts per token from topk to max_select.
    Requires --moe-expert-capacity-factor to be set for dynamic token count support.
    
    Args:
        top_indices: Previously recorded expert indices [num_tokens, topk]
        current_top_indices: Current expert indices [num_tokens, topk]  
        scores: Router scores for all experts [num_tokens, num_experts]
    
    Returns:
        probs: Scores for selected experts [num_tokens, max_select]
        top_indices: Selected expert indices [num_tokens, max_select]
        selected_union_mask: Mask indicating which experts are in union [num_tokens, max_select]
    """
    # Step 1: Compute union of recorded and current expert selections
    union_indices = torch.cat([current_top_indices, top_indices], dim=-1)
    union_mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
    union_mask.scatter_(dim=-1, index=union_indices, value=True)

    num_selects = union_mask.sum(-1)  # [num_tokens], number of experts in union per token
    max_select = num_selects.max().item()  # Maximum union size across all tokens

    # Step 2: Ensure all tokens select exactly max_select experts
    # Use score boosting to guarantee all union experts are selected first,
    # then pad with highest-scoring non-union experts for tokens with union_size < max_select
    masked_scores = scores.clone()
    boost_value = scores.abs().max() + 100.0  # Large enough to ensure union experts rank highest
    masked_scores = torch.where(union_mask, masked_scores + boost_value, masked_scores)
    
    # Select top max_select experts (includes all union experts + padding)
    _, top_indices = torch.topk(masked_scores, k=max_select, dim=1, sorted=False)
    
    # Gather original scores (without boost) for the selected experts
    probs = scores.gather(1, top_indices)
    
    # Zero out probabilities for padding experts (not in union)
    # For tokens where num_selects < max_select, some selected experts are just padding
    # These padding experts should have probability 0
    # Create a mask: True for experts that are in the union, False for padding
    selected_union_mask = union_mask.gather(1, top_indices).float()  # [num_tokens, max_select]
    probs = probs * selected_union_mask  # Zero out padding experts
    
    # Sanity check: detect NaN/Inf early (only in debug mode)
    # if torch.isnan(probs).any() or torch.isinf(probs).any():
    #     import torch.distributed as dist
    #     rank = dist.get_rank() if dist.is_initialized() else 0
    #     print(f"[WARNING] Rank {rank}: Detected NaN/Inf in routing probs after combine_topk_indices!")
    #     print(f"  probs stats: min={probs.min()}, max={probs.max()}, nan={torch.isnan(probs).sum()}, inf={torch.isinf(probs).sum()}")
    
    # CRITICAL: Return the selected_union_mask for later use in replay stages
    return probs, top_indices, selected_union_mask


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]

            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            elif routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                # Record both routing decisions and scores for debugging
                ROUTING_REPLAY.record(top_indices, scores=scores)

            elif routing_replay_stage == "union":
                # In union mode, we calculate the topk indices as usual,
                # However, we will combine them with recorded indices for this forward, and update the recorded indices.
                # This can be treated as "replay_forward" + "record" in one pass.
                recorded_top_indices, _ = ROUTING_REPLAY.pop_forward()  # Get recorded indices (union_mask is None in record stage)
                _, current_top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                probs, top_indices, selected_union_mask = combine_topk_indices(recorded_top_indices, current_top_indices, scores)
                # Logging for debugging
                if top_indices.shape[-1] != topk:
                    import torch.distributed as dist
                    if ROUTING_REPLAY.forward_index == 1:  # Only log once (after first pop)
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        if rank == 0:
                            print(f"[RoutingReplay Union] topk {topk} → {top_indices.shape[-1]} experts per token")
                # Update recorded indices with union results
                ROUTING_REPLAY.update(top_indices, ROUTING_REPLAY.forward_index - 1, selected_union_mask, scores=scores)  # index should -1 as we have updated forward_index

            elif routing_replay_stage == "replay_forward":
                top_indices, selected_union_mask = ROUTING_REPLAY.pop_forward()
                if selected_union_mask is not None:
                    # If selected_union_mask is provided, we need to mask the probs accordingly
                    # And the top_indices may have more than topk experts per token
                    probs = scores.gather(1, top_indices) * selected_union_mask  # First gather, then mask
                else:
                    assert (
                        top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                    ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                    probs = scores.gather(1, top_indices)

            elif routing_replay_stage == "replay_backward":
                top_indices, selected_union_mask = ROUTING_REPLAY.pop_backward()
                if selected_union_mask is not None:
                    # If selected_union_mask is provided, we need to mask the probs accordingly
                    # And the top_indices may have more than topk experts per token
                    probs = scores.gather(1, top_indices) * selected_union_mask  # First gather, then mask
                else:
                    assert (
                        top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                    ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                    probs = scores.gather(1, top_indices)

            else:
                raise ValueError(f"Unknown ROUTING_REPLAY_STAGE: {routing_replay_stage}")

            return probs, top_indices

        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
        print(f"[RoutingReplay] Registered routing replay for module {module}")
