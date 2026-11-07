import re
import sys

LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
EXPERT = re.compile(r"(model\.language_model\.layers)\.(\d+)\.mlp\.experts\.(\d+)\.")
PLE_TABLE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")
# like vLLM's own EP filter: only the large tensors of the other EP rank's experts are skipped, never their scales
HEAVY = (".qweight", ".weight", ".weight_packed")
TARGET = "Qwen4ExpForConditionalGeneration"


def rule(start: int, end: int, local: set[int] | None, positions: dict | None, exempt: frozenset = frozenset()):
    def skip(name: str) -> bool:
        if name in exempt:
            return False
        layer = LAYER.match(name)
        if layer is None:
            return False
        if not start <= int(layer.group(1)) < end or PLE_TABLE.search(name):
            return True
        expert = EXPERT.match(name)
        if expert is None:
            return False
        number = int(expert.group(3))
        if positions is not None:
            number = positions.get((expert.group(1), int(expert.group(2))), {}).get(number)
            if number is None:  # pruned
                return True
        return local is not None and number not in local and name.endswith(HEAVY)

    return skip


def ple_only(name: str) -> bool:
    return ".ple." not in name


def current_rule(keep: str | None):
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.ple_offload_layer import is_offload_process

    if is_offload_process():
        print("read filter: PLE offload worker, PLE tensors only", file=sys.stderr, flush=True)
        return ple_only
    config = get_current_vllm_config()
    if TARGET not in (config.model_config.hf_config.architectures or []):
        return None
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.model_loader.ep_weight_filter import compute_local_expert_ids

    parallel = config.parallel_config
    pp = get_pp_group()
    start, end = get_pp_indices(config.model_config.hf_text_config.num_hidden_layers, pp.rank_in_group, pp.world_size)
    local = None
    if parallel.enable_expert_parallel and not parallel.enable_eplb and parallel.data_parallel_size == 1 \
            and parallel.prefill_context_parallel_size == 1:  # ep_rank = tp_rank, as in DefaultModelLoader._init_ep_weight_filter
        local = compute_local_expert_ids(config.model_config.get_num_experts(), parallel.tensor_parallel_size,
                                         get_tensor_model_parallel_rank(), placement=parallel.expert_placement_strategy)
    print(f"read filter: layers [{start}, {end}), {len(local) if local is not None else 'all'} local experts, "
          f"REAP keep {'on' if keep else 'off'}", file=sys.stderr, flush=True)
    from arc_reap import positions, read_keep

    return rule(start, end, local, positions(read_keep(keep)) if keep else None, override_names())


def override_names() -> frozenset:
    import os

    path = os.environ.get("ARC_VLLM_WEIGHT_OVERRIDE")
    if not path:
        return frozenset()
    from safetensors import safe_open

    with safe_open(path, framework="pt") as stream:
        return frozenset(stream.keys())


def patch_skip(module, keep: str | None) -> None:
    original, rules = module.should_skip_weight, {}

    def should_skip_weight(weight_name, local_expert_ids):
        if original(weight_name, local_expert_ids):
            return True
        from vllm.config import get_current_vllm_config

        key = id(get_current_vllm_config().model_config)
        if key not in rules:
            rules[key] = current_rule(keep)
        return rules[key] is not None and rules[key](weight_name)

    module.should_skip_weight = should_skip_weight


def patches(keep: str | None) -> dict:
    return {"vllm.model_executor.model_loader.weight_utils": lambda module: patch_skip(module, keep)}

import os

from arc_patches import chain
if os.environ.get("ARC_VLLM_READ_FILTER") == "1":
    for _name, _hook in patches(os.environ.get("ARC_VLLM_REAP_KEEP")).items():
        chain(_name, _hook)
