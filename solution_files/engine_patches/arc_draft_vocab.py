import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("vllm_draft_vocab")


class DraftVocab:

    def __init__(self, weight, ids, pad, vocab: int, gather):
        self.weight, self.ids, self.pad, self.vocab, self.gather = weight, ids, pad, vocab, gather

    @classmethod
    def build(cls, head, token_ids: list[int]):
        weight = getattr(head, "weight", None)
        if weight is None or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            logger.warning("draft vocab: the drafter's LM head is quantized or has no plain weight; keeping the full head")
            return None
        tp, rank = head.tp_size, head.tp_rank
        per = head.org_vocab_size_padded // tp
        ranges = [(r * per, min((r + 1) * per, head.org_vocab_size)) for r in range(tp)]
        indices = getattr(head, "shard_indices", None)
        if indices is not None and (indices.org_vocab_start_index, indices.org_vocab_end_index) != ranges[rank]:
            logger.warning("draft vocab: unexpected LM-head sharding %s; keeping the full head", ranges[rank])
            return None
        ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
        ids = ids[ids < head.org_vocab_size]
        shards = [ids[(ids >= start) & (ids < end)] for start, end in ranges]
        m = max(len(shard) for shard in shards)
        start = ranges[rank][0]
        local = torch.zeros(m, weight.shape[1], dtype=weight.dtype, device=weight.device)
        local[:len(shards[rank])] = weight[(shards[rank] - start).to(weight.device)]
        if tp == 1:
            gather = None
        elif getattr(head, "parallel_group", None) is not None:
            gather = lambda x, dim=-1: head.parallel_group.all_gather(x, dim=dim)
        else:
            from vllm.distributed import tensor_model_parallel_all_gather
            gather = lambda x, dim=-1: tensor_model_parallel_all_gather(x, dim=dim)
        columns = torch.zeros(tp * m, dtype=torch.long)
        pad = torch.ones(tp * m, dtype=torch.bool)
        for r, shard in enumerate(shards):
            columns[r * m:r * m + len(shard)] = shard
            pad[r * m:r * m + len(shard)] = False
        logger.info("draft vocab: %d of %d tokens, %d rows per rank", len(ids), head.org_vocab_size, m)
        return cls(local, columns.to(weight.device), pad.to(weight.device), head.org_vocab_size, gather)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = hidden.to(self.weight.dtype) @ self.weight.t()
        if self.gather is not None:
            logits = self.gather(logits)
        return logits.masked_fill(self.pad, float("-inf"))

    def full_probs(self, probs: torch.Tensor) -> torch.Tensor:
        full = probs.new_zeros(probs.shape[0], self.vocab)
        return full.scatter_add_(1, self.ids.expand(probs.shape[0], -1), probs)


def restricted_logits(state: DraftVocab):
    keep = (~state.pad).nonzero().flatten()
    ids = state.ids.index_select(0, keep)

    def compute_logits(hidden_states, *args, **kwargs):
        logits = state.logits(hidden_states)
        full = logits.new_full((logits.shape[0], state.vocab), float("-inf"))
        return full.index_copy_(1, ids, logits.index_select(1, keep))

    return compute_logits


def patch_v2_speculator(module, path: str) -> None:
    token_ids = json.loads(Path(path).read_text())["token_ids"]
    cls = module.DraftModelSpeculator
    load = cls.load_model

    def load_model(self, target_model):
        load(self, target_model)
        if getattr(self.vllm_config.speculative_config, "method", None) != "mtp":
            return
        state = DraftVocab.build(self.model.lm_head, token_ids)
        if state is not None:
            self.model.compute_logits = restricted_logits(state)

    cls.load_model = load_model


def patch_proposer(module, path: str) -> None:
    token_ids = json.loads(Path(path).read_text())["token_ids"]
    cls = getattr(module, "LLMBaseProposer", None)
    if cls is None:
        return
    greedy, sample = cls._greedy_sample, cls._sample_draft_tokens

    def vocab(self):
        state = self.__dict__.get("_arc_draft_vocab", False)
        if state is False:
            usable = self.method == "mtp" and not getattr(self, "use_heterogeneous_vocab", False)
            state = DraftVocab.build(self.model.lm_head, token_ids) if usable else None
            self._arc_draft_vocab = state
        return state

    def _greedy_sample(self, hidden_states):
        state = vocab(self)
        if state is None:
            return greedy(self, hidden_states)
        return state.ids[state.logits(hidden_states).argmax(dim=-1)]

    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        state = vocab(self)
        if state is None:
            return sample(self, hidden_states, sampling_metadata)
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        columns, probs = self._sample_from_logits(state.logits(hidden_states), sampling_metadata)
        return state.ids[columns], (None if probs is None else state.full_probs(probs))

    cls._greedy_sample = _greedy_sample
    cls._sample_draft_tokens = _sample_draft_tokens


def patches(path: str) -> dict:
    return {"vllm.v1.spec_decode.llm_base_proposer": lambda module: patch_proposer(module, path),
            "vllm.v1.worker.gpu.spec_decode.speculator": lambda module: patch_v2_speculator(module, path)}

import os

from arc_patches import PATCHES
if os.environ.get("ARC_VLLM_DRAFT_VOCAB"):
    PATCHES.update(patches(os.environ["ARC_VLLM_DRAFT_VOCAB"]))
