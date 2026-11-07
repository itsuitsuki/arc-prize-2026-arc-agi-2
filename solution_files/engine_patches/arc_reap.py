import hashlib
import json
import os
import re
from pathlib import Path

EXPERT = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.mlp\.gate\.weight$")
TARGET, DRAFT = "model.language_model.layers", "mtp.layers"


def read_keep(path) -> dict[tuple[str, int], list[int]]:
    return {(key.rsplit(".", 1)[0], int(key.rsplit(".", 1)[1])): experts for key, experts in json.loads(Path(path).read_text()).items()}


def positions(kept: dict[tuple[str, int], list[int]]) -> dict:
    return {key: {expert: new for new, expert in enumerate(experts)} for key, experts in kept.items()}


def keep_sha256(keep: Path) -> str:
    canonical = json.dumps(json.loads(Path(keep).read_text()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def renamed(name: str, kept: dict[tuple[str, int], list[int]], position: dict) -> str | None:
    match = EXPERT.match(name)
    if not match:
        return name
    prefix, layer, expert, rest = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    if (prefix, layer) not in kept:
        raise KeyError(f"no REAP statistics for {prefix}.{layer}")
    new = position[(prefix, layer)].get(expert)
    return None if new is None else f"{prefix}.{layer}.mlp.experts.{new}.{rest}"


def patch_reap_keep(module, path: str, class_name: str) -> None:
    import torch

    kept = read_keep(path)
    position = positions(kept)
    cls = getattr(module, class_name)
    original = cls.load_weights

    def load_weights(self, weights):
        def pruned():
            for name, tensor in weights:
                router = ROUTER.match(name)
                if router:
                    yield name, tensor[torch.tensor(kept[(router.group(1), int(router.group(2)))])].contiguous()
                    continue
                new = renamed(name, kept, position)
                if new is not None:
                    yield new, tensor

        return original(self, pruned())

    cls.load_weights = load_weights


def patch_marlin_meta_workspace(module) -> None:
    import torch

    make = module.marlin_make_workspace_new

    def make_workspace(device, *args, **kwargs):
        if torch.device(device).type == "meta":
            return torch.empty(0, dtype=torch.int, device="meta")
        return make(device, *args, **kwargs)

    module.marlin_make_workspace_new = make_workspace

from arc_patches import PATCHES, chain
if os.environ.get("ARC_VLLM_REAP_KEEP"):
    for _module, _class in (("vllm.models.qwen4_exp.nvidia.model", "Qwen4ExpForConditionalGeneration"), ("vllm.models.qwen4_exp.nvidia.mtp", "Qwen4ExpMTP")):
        chain(_module, lambda module, name=_class: patch_reap_keep(module, os.environ["ARC_VLLM_REAP_KEEP"], name))
    PATCHES["vllm.model_executor.layers.quantization.utils.marlin_utils"] = patch_marlin_meta_workspace
