from dataclasses import replace

_STATS = {"reused": 0}


def patch_gdn_recoverssm(module) -> None:
    cls, base = module.GDNRecoverSSMMetadataBuilder, module.GDNAttentionMetadataBuilder
    init = cls.__init__

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        init(self, kv_cache_spec, layer_names, vllm_config, device)
        # the stock GDN builder's condition, without its exact-type check
        self.supports_update_block_table = vllm_config.use_v2_model_runner and device.type == "cuda"
        if self.supports_update_block_table:
            self.mamba_aligned_state_indices = None  # opts into the V2 model runner's aligned state-index precompute

    def update_block_table(self, metadata, blk_table, slot_mapping):
        meta = base.update_block_table(self, metadata, blk_table, slot_mapping)
        source = metadata.recoverssm_commit
        if source is None:
            meta = replace(meta, recoverssm_commit=None, recoverssm_context=None)
        else:
            indices = meta.spec_state_indices_tensor[:, :1]  # the checkpoint column, which build() also keeps
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            commit = replace(source, state_indices=indices, block_table=blk_table if align else None)
            meta = replace(meta, spec_state_indices_tensor=indices, recoverssm_commit=commit,
                           recoverssm_context=self._get_context())
        _STATS["reused"] += 1
        if _STATS["reused"] == 1:
            module.logger.info("ARC_VLLM_GDN_REUSE: GDN RecoverSSM groups share one metadata build per step")
        return meta

    cls.__init__ = __init__
    cls.update_block_table = update_block_table


def patches() -> dict:
    return {"vllm.v1.attention.backends.gdn_recoverssm": patch_gdn_recoverssm}

import os

from arc_patches import chain
if os.environ.get("ARC_VLLM_GDN_REUSE") == "1":
    for _name, _hook in patches().items():
        chain(_name, _hook)
