"""SGLang router statistics for REAP expert pruning, excerpted from the engine patches the statistics run used.

With ARC_SGLANG_ROUTER_STATS=<file> (an SGLang server run for REAP expert pruning, CUDA graphs off so the hooks run)
every MoE TopK accumulates, per expert, how often it is routed, the sum of its routing weights, and the sum of the
norms of its routing-weighted outputs g_j(x) f_j(x) (REAP's saliency numerator), read from the Triton MoE's
[tokens, top_k, hidden] buffer just before the experts are summed; under TP each rank sees its partial f_j(x). TopK
modules are numbered in construction order (the target's layers first, then the MTP layer), and every TP rank writes
{index: (layer_id, tensor[3, experts])} to <file>.rank<R> every ROUTER_STATS_EVERY calls of the first TopK.
"""
import importlib.abc
import importlib.util
import math
import os
import sys
from collections import defaultdict


ROUTER_STATS_EVERY = 256


_ROUTER = {"modules": [], "stats": {}, "calls": 0, "last": None}


def patch_router_stats(module, path: str) -> None:
    import torch

    topk = module.TopK
    init = topk.__init__
    modules, stats = _ROUTER["modules"], _ROUTER["stats"]

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._arc_index = len(modules)
        modules.append(self)

    def record(self, out) -> None:
        ids, weights, logits = out.topk_ids, out.topk_weights, out.router_logits
        experts = logits.shape[-1]
        table = stats.get(self._arc_index)
        if table is None:
            table = stats[self._arc_index] = torch.zeros(3, experts, dtype=torch.float32, device=ids.device)
        valid = (ids >= 0) & (ids < experts)
        flat = ids.clamp(0, experts - 1).reshape(-1).long()
        table[0].index_add_(0, flat, valid.reshape(-1).float())
        table[1].index_add_(0, flat, (weights.float() * valid).reshape(-1))
        _ROUTER["last"] = (table, flat, valid.reshape(-1).float(), ids.shape)
        if self._arc_index == 0:
            _ROUTER["calls"] += 1
            if _ROUTER["calls"] % ROUTER_STATS_EVERY == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                target = f"{path}.rank{rank}"
                torch.save({index: (modules[index].layer_id, table.cpu()) for index, table in stats.items()}, target + ".part")
                os.replace(target + ".part", target)

    for name in ("forward_cuda", "forward_native"):
        original = getattr(topk, name)

        def wrapped(self, *args, _original=original, **kwargs):
            out = _original(self, *args, **kwargs)
            record(self, out)
            return out

        setattr(topk, name, wrapped)
    topk.__init__ = __init__


def patch_expert_output_norms(module) -> None:
    """Wrap the Triton MoE's sum over the top-k experts: add each slot's output norm to the expert routed there."""
    for name in ("moe_sum_reduce", "moe_sum_reduce_triton", "moe_sum_reduce_torch_compile"):
        original = getattr(module, name, None)
        if original is None:
            continue

        def wrapped(x, *args, _original=original, **kwargs):
            last = _ROUTER["last"]
            if last is not None and tuple(x.shape[:2]) == tuple(last[3]):
                table, flat, valid, _ = last
                table[2].index_add_(0, flat, x.float().norm(dim=-1).reshape(-1) * valid)
            return _original(x, *args, **kwargs)

        setattr(module, name, wrapped)



_PATCHES = {}
if os.environ.get("ARC_SGLANG_ROUTER_STATS"):
    _PATCHES["sglang.srt.layers.moe.topk"] = lambda module: patch_router_stats(module, os.environ["ARC_SGLANG_ROUTER_STATS"])
    _PATCHES["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"] = patch_expert_output_norms


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in _PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            _PATCHES[name](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Hook())
