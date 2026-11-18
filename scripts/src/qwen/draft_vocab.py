"""Frequency-ranked draft vocabulary (FR-Spec, Zhao et al. 2025): the K most frequent completion tokens of a set of
traces, which src/qwen/vllm_patch/vllm_draft_vocab.py lets the MTP drafter score instead of the whole 248K vocabulary.
Verification stays on the full vocabulary, so the output distribution is unchanged; only acceptance can drop, by the
share of target tokens outside the set.
    python -m src.qwen.draft_vocab --results DIR [DIR ...] --top 32768 --output vocab.json [--check DIR ...]
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def count(directories: list[Path]) -> Counter:
    counts = Counter()
    for directory in directories:
        for path in directory.glob("*.json"):
            counts.update(json.loads(path.read_text()).get("completion_token_ids") or [])
    return counts


def coverage(counts: Counter, keep: set[int]) -> float:
    total = sum(counts.values())
    return sum(n for token, n in counts.items() if token in keep) / max(total, 1)


def top_tokens(counts: Counter, k: int, always: list[int]) -> list[int]:
    """The k most frequent ids plus the always-kept ones (special tokens such as </think> and <|im_end|>), sorted."""
    return sorted(set(always) | {token for token, _ in counts.most_common(k - len(set(always)))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="result dirs with completion_token_ids")
    parser.add_argument("--top", type=int, default=32768)
    parser.add_argument("--always", type=int, nargs="*", default=[], help="token ids kept regardless of frequency")
    parser.add_argument("--check", type=Path, nargs="*", default=[], help="held-out result dirs to report coverage on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = count(args.results)
    ids = top_tokens(counts, args.top, args.always)
    keep = set(ids)
    report = {"token_ids": ids, "k": len(ids), "tokens_counted": sum(counts.values()), "distinct": len(counts),
              "coverage": round(coverage(counts, keep), 6)}
    if args.check:
        report["held_out_coverage"] = round(coverage(count(args.check), keep), 6)
    args.output.write_text(json.dumps(report))
    print(json.dumps({key: value for key, value in report.items() if key != "token_ids"}))
