import os


def patch_qwen4_exp_topk_logits(module, k: int) -> None:
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

    def compute_logits(self, hidden_states):
        processor, head = self.logits_processor, self.lm_head
        assert processor.soft_cap is None and processor.scale == 1.0 and not processor.logits_as_input
        local = processor._apply_head(head, hidden_states, None)
        padding = head.shard_indices.num_org_vocab_padding
        if padding:
            local[..., -padding:] = float("-inf")
        values, ids = local.topk(k, dim=-1)
        ids = ids + head.shard_indices.org_vocab_start_index
        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values.contiguous(), dim=-1)
            ids = tensor_model_parallel_all_gather(ids.contiguous(), dim=-1)
        rows = torch.full((local.shape[0], processor.org_vocab_size), float("-inf"), dtype=local.dtype, device=local.device)
        return rows.scatter_(-1, ids, values)

    module.Qwen4ExpForCausalLM.compute_logits = compute_logits

from arc_patches import chain
if os.environ.get("ARC_VLLM_TOPK_LOGITS"):  # after the REAP hook on the same module
    chain("vllm.models.qwen4_exp.nvidia.model", lambda module, k=int(os.environ["ARC_VLLM_TOPK_LOGITS"]): patch_qwen4_exp_topk_logits(module, k))
