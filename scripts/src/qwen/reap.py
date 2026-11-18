"""REAP expert pruning (Router-weighted Expert Activation Pruning, Lasby et al. 2025) for Qwen3.8-Flash-Next checkpoints.

An expert's REAP saliency is the mean, over the tokens routed to it, of g_j(x) * ||f_j(x)||: its routing weight times
the norm of its output. The statistics come from an SGLang run with ARC_SGLANG_ROUTER_STATS (src/qwen/vllm_patch): every
TP rank writes {topk index: (layer_id, tensor[3, experts])} with the routed count, the routing-weight sum and the sum of
||g_j(x) f_j(x)|| per expert; under TP a rank sees its share of f_j(x), so the ranks' saliencies are averaged. TopK
indices follow construction order: the target's layers, then the MTP layer.

Every MoE layer (and the MTP layer) keeps its `keep` highest-saliency experts in their original order, renumbered from
0; its router rows are sliced to match. Shards holding expert or router tensors are rewritten, every other file is
symlinked to the source, and config.json gets num_experts = keep.
    python -m src.qwen.reap --stats STATS.pt.rank0 [STATS.pt.rank1 ...] --source MODEL --output OUT --keep 384
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

EXPERT = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.(\d+)\.mlp\.gate\.weight$")
TARGET, DRAFT = "model.language_model.layers", "mtp.layers"


def keep_sha256(keep: Path) -> str:
    """sha256 of the keep list's canonical JSON (sorted keys, no spaces): reformatting the file does not change it."""
    canonical = json.dumps(json.loads(Path(keep).read_text()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def saliency(stats_files: list[Path], target_layers: int) -> dict[tuple[str, int], "torch.Tensor"]:
    """(prefix, layer) -> REAP saliency per expert, averaged over the ranks' files."""
    import torch

    per_rank = [torch.load(path) for path in stats_files]
    scores = {}
    for index in per_rank[0]:
        layer_id, _ = per_rank[0][index]
        key = (TARGET, layer_id) if index < target_layers else (DRAFT, index - target_layers)
        rank_scores = []
        for stats in per_rank:
            table = stats[index][1].double()
            rank_scores.append(torch.where(table[0] > 0, table[2] / table[0].clamp(min=1), torch.zeros_like(table[0])))
        scores[key] = torch.stack(rank_scores).mean(0)
    return scores


def keep_sets(scores: dict, keep: int) -> dict[tuple[str, int], list[int]]:
    return {key: sorted(score.argsort(descending=True)[:keep].tolist()) for key, score in scores.items()}


def renamed(name: str, kept: dict[tuple[str, int], list[int]], position: dict) -> str | None:
    """The tensor's name in the pruned checkpoint, or None when its expert is pruned."""
    match = EXPERT.match(name)
    if not match:
        return name
    prefix, layer, expert, rest = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    if (prefix, layer) not in kept:
        raise KeyError(f"no REAP statistics for {prefix}.{layer}")
    new = position[(prefix, layer)].get(expert)
    return None if new is None else f"{prefix}.{layer}.mlp.experts.{new}.{rest}"


def prune(source: Path, output: Path, kept: dict[tuple[str, int], list[int]]) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    position = {key: {expert: new for new, expert in enumerate(experts)} for key, experts in kept.items()}
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    output.mkdir(parents=True, exist_ok=False)
    touched = {file for name, file in weight_map.items() if EXPERT.match(name) or ROUTER.match(name)}
    new_map, dropped = {}, 0
    for file in sorted(set(weight_map.values())):
        names = [name for name, shard in weight_map.items() if shard == file]
        if file not in touched:
            os.symlink(os.path.realpath(source / file), output / file)
            new_map.update({name: file for name in names})
            continue
        tensors = {}
        with safe_open(str(source / file), framework="pt") as stream:
            for name in names:
                router = ROUTER.match(name)
                new = renamed(name, kept, position)
                if new is None:
                    dropped += 1
                    continue
                tensor = stream.get_tensor(name)
                if router:
                    tensor = tensor[torch.tensor(kept[(router.group(1), int(router.group(2)))])].contiguous()
                tensors[new] = tensor
        save_file(tensors, str(output / file), metadata={"format": "pt"})
        new_map.update({name: file for name in tensors})
    (output / "model.safetensors.index.json").write_text(json.dumps({**index, "weight_map": new_map}, indent=2))
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    experts = {len(v) for v in kept.values()}
    if len(experts) != 1:
        raise ValueError(f"layers keep different expert counts: {experts}")
    text["num_experts"] = experts.pop()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for entry in source.iterdir():  # tokenizer, templates, n-gram table shards outside the index, runtime files
        if not (output / entry.name).exists():
            os.symlink(os.path.realpath(entry), output / entry.name)
    return {"rewritten_files": len(touched), "dropped_tensors": dropped, "num_experts": text["num_experts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, nargs="+", required=True, help="ARC_SGLANG_ROUTER_STATS files, one per TP rank")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, required=True, help="experts kept per MoE layer (Flash-Next has 512)")
    parser.add_argument("--target-layers", type=int, default=48, help="TopK modules of the target before the MTP layer")
    args = parser.parse_args()
    kept = keep_sets(saliency(args.stats, args.target_layers), args.keep)
    (args.output.parent / f"{args.output.name}.reap_keep.json").write_text(
        json.dumps({f"{prefix}.{layer}": experts for (prefix, layer), experts in kept.items()}))
    print(json.dumps(prune(args.source, args.output, kept)))


if __name__ == "__main__":
    main()
