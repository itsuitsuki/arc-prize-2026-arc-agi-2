"""Post-trained Qwen3.8-Flash-Next variants that could replace the base model in the deployment, and a tensor-level
diff against a reference checkpoint.

The diff decides how a variant can be deployed: if only non-expert tensors changed, they can ship as a
--weight-override on top of the existing REAP-448 W4A16 deployment (kaggle/qwen-transductive/build_notebook.py); if
routed experts or the router changed, the REAP keep list has to be recomputed and the experts re-quantized.

Every *.safetensors file under a directory is read through its header (8-byte little-endian length + JSON), so the
n-gram (PLE) shards, MTP files and extra-tensor files that a sharded index may leave out are compared too. Content is
compared by a hash of each tensor's raw bytes; tensors present in both with the same dtype and shape but different
bytes are "changed".

override builds that --weight-override file for a variant whose diff touched only non-expert tensors: the changed
names (a Merlin checkpoint's packed INT8 names map back to the BF16 .weight they quantize) are fetched from the
variant's BF16 Hugging Face repo by HTTP range requests, tensor by tensor, so a ~5 GB override needs no 360 GB
download. Every tensor must exist in the deployed checkpoint (--deploy) with the same dtype and shape, as
src.qwen.fnweights.patch_weight_override requires, and the file carries the deployed REAP keep list's sha256.
--without-experts leaves a variant's routed-expert changes out (the deployed int4 experts stay) for a variant whose
expert deltas are small next to the int4 step; a changed router is still refused.

  python -m src.qwen.fn_variants diff --a VARIANT_DIR --b REFERENCE_DIR --out diff.json [--no-hash]
  python -m src.qwen.fn_variants override --diff diff.json --repo OWNER/NAME --deploy DEPLOYED_DIR --keep KEEP.json \
      --out override.safetensors [--revision main] [--without-experts]
  python -m src.qwen.fn_variants selftest
"""
import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

from src.qwen.reap import keep_sha256
from src.utils.tensors import safetensors_header

# name -> (Hugging Face repo, local directory name, format, reference to diff against, note)
VARIANTS = {
    "swift15-merlin": ("halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin", "Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin",
                       "merlin-w4a16", "base-merlin",
                       "UkisAI Swift 1.5 (RL against overthinking; card: -63.4% thinking tokens on GPQA, <1% loss at "
                       "xhigh) in halt95's Merlin W4A16 layout; Swift Open License v1.0 + Qwen Community License 1.0"),
    "base-merlin": ("halt95/Qwen3.8-Flash-Next-W4A16-Merlin", "Qwen3.8-Flash-Next-W4A16-Merlin", "merlin-w4a16", None,
                    "the base model in the same Merlin layout, the reference for swift15-merlin"),
    "agnes30": ("Agnes-AI/Agnes-3.0-Qwen", "Agnes-3.0-Qwen", "bf16", "base-bf16",
                "Agnes AI post-trained Flash-Next (method undisclosed; self-reported GPQA 91.1, Terminal-Bench 2.1 "
                "87.8); Qwen Community License 1.0"),
    "base-bf16": ("Qwen/Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-BF16", "bf16", None,
                  "the base model in BF16, the reference for agnes30"),
}

CATEGORIES = (  # first match wins
    ("mtp", r"(^|\.)mtp\.|nextn"),
    ("ngram_ple", r"ngram|engram|ple_|per_layer_emb"),
    ("router", r"mlp\.gate\.(weight|bias)$|router|e_score_correction"),
    ("shared_expert", r"shared_expert"),
    ("routed_expert", r"mlp\.experts\.|\.experts\."),
    ("gdn", r"linear_attn|delta|\.conv1d|A_log|dt_bias"),
    ("attention", r"self_attn|\.attn\.|q_proj|k_proj|v_proj|o_proj|indexer|qsa"),
    ("embed_lm_head", r"embed_tokens|lm_head"),
    ("norm", r"norm"),
    ("vision_audio", r"visual|vision|audio"),
)


def category(name):
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "other"


def inventory(directory, hash_content=True):
    """name -> {"dtype", "shape", "file", "sha1"?} for every tensor in every *.safetensors under directory."""
    tensors = {}
    for path in sorted(Path(directory).rglob("*.safetensors")):
        header, base = safetensors_header(path)
        with open(path, "rb") as handle:
            for name, info in header.items():
                entry = {"dtype": info["dtype"], "shape": info["shape"], "file": path.name}
                if hash_content:
                    start, end = info["data_offsets"]
                    handle.seek(base + start)
                    digest, left = hashlib.sha1(), end - start
                    while left:
                        block = handle.read(min(left, 64 << 20))
                        digest.update(block)
                        left -= len(block)
                    entry["sha1"] = digest.hexdigest()
                # a name in two files (e.g. lm_head in the main shards and in an MTP file) keeps both, the later one
                # under name@file
                tensors[f"{name}@{path.name}" if name in tensors else name] = entry
    return tensors


def diff(variant, reference):
    """Per category: tensors only in the variant, only in the reference, with a different dtype/shape, and with the
    same dtype/shape but different content (only when both inventories carry hashes)."""
    names = set(variant) | set(reference)
    report = {"by_category": {}, "only_variant": [], "only_reference": [], "layout_changed": [], "content_changed": []}
    for name in sorted(names):
        a, b = variant.get(name), reference.get(name)
        if b is None:
            kind = "only_variant"
        elif a is None:
            kind = "only_reference"
        elif (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            kind = "layout_changed"
        elif "sha1" in a and "sha1" in b and a["sha1"] != b["sha1"]:
            kind = "content_changed"
        else:
            kind = "same"
        cat = report["by_category"].setdefault(category(name), Counter())
        cat[kind] += 1
        if kind != "same":
            report[kind].append(name)
    report["by_category"] = {k: dict(v) for k, v in sorted(report["by_category"].items())}
    report["counts"] = {k: len(report[k]) for k in ("only_variant", "only_reference", "layout_changed", "content_changed")}
    return report


MERLIN_SUFFIX = re.compile(r"\.weight_(packed|scale|shape)$")


def bf16_names(changed):
    """The BF16 checkpoint names behind a diff's changed names (packed INT8 triples collapse to one .weight)."""
    return sorted({MERLIN_SUFFIX.sub(".weight", name.split("@")[0]) for name in changed})


class RemoteCheckpoint:
    """Range reads of a Hugging Face safetensors checkpoint through HfFileSystem (HF_ENDPOINT and the token apply): the
    index, each file's header, then single tensors. open_file(name) -> binary file object replaces the Hub in tests."""

    def __init__(self, repo, revision="main", open_file=None):
        if open_file is None:
            from huggingface_hub import HfFileSystem

            filesystem = HfFileSystem()
            open_file = lambda name: filesystem.open(f"{repo}/{name}", "rb", revision=revision)  # noqa: E731
        self.open_file = open_file
        self.headers = {}
        with self.open_file("model.safetensors.index.json") as index:
            self.weight_map = json.loads(index.read())["weight_map"]

    def read(self, file, start, end):
        with self.open_file(file) as stream:
            stream.seek(start)
            data = stream.read(end - start)
        if len(data) != end - start:
            raise IOError(f"{file} bytes {start}-{end}: got {len(data)} bytes")
        return data

    def header(self, file):
        if file not in self.headers:
            with self.open_file(file) as stream:
                self.headers[file] = safetensors_header(stream)
        return self.headers[file]

    def info(self, name):
        header, base = self.header(self.weight_map[name])
        return header[name], base

    def tensor(self, name):
        info, base = self.info(name)
        start, end = info["data_offsets"]
        return self.read(self.weight_map[name], base + start, base + end)


def build_override(changed, remote, deployed, keep, out, source):
    """Write override.safetensors with the BF16 tensors behind `changed`, checked against the deployed inventory."""
    names = bf16_names(changed)
    entries, offset = {}, 0
    for name in names:
        info, _ = remote.info(name)
        have = deployed.get(name)
        if have is None or (have["dtype"], have["shape"]) != (info["dtype"], info["shape"]):
            raise ValueError(f"{name}: variant {info['dtype']} {info['shape']}, deployed "
                             f"{have and (have['dtype'], have['shape'])}")
        size = info["data_offsets"][1] - info["data_offsets"][0]
        entries[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    header = {"__metadata__": {"reap_keep_sha256": keep_sha256(keep), "source": source, "tensors": str(len(names))},
              **entries}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # align the data section as safetensors writers do
    temporary = Path(f"{out}.part")
    with open(temporary, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw)
        for name in names:
            handle.write(remote.tensor(name))
    temporary.replace(out)
    return {"tensors": len(names), "bytes": offset, "by_category": dict(Counter(category(n) for n in names))}


def override_names(report, without_experts=False):
    """(names to fetch, routed-expert names left out); raises SystemExit when the override cannot carry the diff."""
    changed = report["content_changed"] + report["layout_changed"]
    experts = [n for n in changed if category(n) == "routed_expert"]
    if any(category(n) == "router" for n in changed) or (experts and not without_experts):
        raise SystemExit("the variant changed routers or routed experts: an override cannot carry it")
    return [n for n in report["content_changed"] if category(n) != "routed_expert"], experts


def selftest():
    import tempfile

    def write(path, tensors):
        header, blob = {}, b""
        for name, (dtype, shape, data) in tensors.items():
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
            blob += data
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)

    with tempfile.TemporaryDirectory() as scratch:
        a, b = Path(scratch, "a"), Path(scratch, "b")
        a.mkdir(), b.mkdir()
        common = {"model.layers.0.mlp.experts.3.down_proj.qweight": ("I32", [2], b"\x01" * 8),
                  "model.layers.0.self_attn.q_proj.weight": ("BF16", [2], b"\x02" * 4)}
        write(b / "model-00001.safetensors", common)
        write(a / "model-00001.safetensors", {**common, "model.layers.0.self_attn.q_proj.weight": ("BF16", [2], b"\x03" * 4)})
        write(a / "model-extra.safetensors", {"model.layers.0.linear_attn.in_proj.weight": ("I8", [4], b"\x04" * 4)})
        write(b / "model-extra.safetensors", {"model.layers.0.linear_attn.in_proj.weight": ("BF16", [4], b"\x04" * 8)})
        report = diff(inventory(a), inventory(b))
    assert report["counts"] == {"only_variant": 0, "only_reference": 0, "layout_changed": 1, "content_changed": 1}, report
    assert report["content_changed"] == ["model.layers.0.self_attn.q_proj.weight"]
    assert report["by_category"]["routed_expert"] == {"same": 1} and report["by_category"]["gdn"] == {"layout_changed": 1}
    assert category("model.layers.3.mlp.gate.weight") == "router" and category("mtp.layers.0.mlp.experts.1.w1") == "mtp"

    # override: a fake Hugging Face that serves two BF16 shards with range support
    changed = ["m.layers.0.linear_attn.in_proj_qkv.weight_packed", "m.layers.0.linear_attn.in_proj_qkv.weight_scale",
               "m.layers.0.mlp.shared_expert.up_proj.weight"]
    assert bf16_names(changed) == ["m.layers.0.linear_attn.in_proj_qkv.weight", "m.layers.0.mlp.shared_expert.up_proj.weight"]
    with tempfile.TemporaryDirectory() as scratch:
        repo, deploy = Path(scratch, "repo"), Path(scratch, "deploy")
        repo.mkdir(), deploy.mkdir()
        qkv, up = bytes(range(8)), bytes(range(10, 14))
        write(repo / "s1.safetensors", {"m.layers.0.linear_attn.in_proj_qkv.weight": ("BF16", [4], qkv)})
        write(repo / "s2.safetensors", {"m.layers.0.mlp.shared_expert.up_proj.weight": ("BF16", [2], up),
                                        "m.layers.0.mlp.experts.gate_up_proj": ("BF16", [2], b"\x00" * 4)})
        (repo / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "m.layers.0.linear_attn.in_proj_qkv.weight": "s1.safetensors",
            "m.layers.0.mlp.shared_expert.up_proj.weight": "s2.safetensors",
            "m.layers.0.mlp.experts.gate_up_proj": "s2.safetensors"}}))
        write(deploy / "d.safetensors", {"m.layers.0.linear_attn.in_proj_qkv.weight": ("BF16", [4], b"\x00" * 8),
                                         "m.layers.0.mlp.shared_expert.up_proj.weight": ("BF16", [2], b"\x00" * 4)})
        keep = Path(scratch, "keep.json")
        keep.write_text('{"0": [1, 2]}')

        remote = RemoteCheckpoint("x/y", open_file=lambda name: open(repo / name, "rb"))
        out = Path(scratch, "override.safetensors")
        summary = build_override(changed, remote, inventory(deploy, False), keep, out, "x/y@main")
        header, base = safetensors_header(out)
        blob = out.read_bytes()[base:]
        assert summary["tensors"] == 2 and summary["bytes"] == 12, summary
        start, end = header["m.layers.0.linear_attn.in_proj_qkv.weight"]["data_offsets"]
        assert blob[start:end] == qkv and blob[end:] == up
        assert json.loads(out.read_bytes()[8:base])["__metadata__"]["reap_keep_sha256"] == keep_sha256(keep)
        try:  # a tensor whose layout differs from the deployed checkpoint is refused
            build_override(changed, remote, {**inventory(deploy, False), "m.layers.0.mlp.shared_expert.up_proj.weight":
                                             {"dtype": "BF16", "shape": [3]}}, keep, out, "x/y@main")
            raise AssertionError("a layout mismatch was accepted")
        except ValueError:
            pass
    report = {"content_changed": ["m.layers.0.mlp.experts.down_proj", "m.layers.0.self_attn.q_proj.weight"], "layout_changed": []}
    assert override_names(report, True) == (["m.layers.0.self_attn.q_proj.weight"], ["m.layers.0.mlp.experts.down_proj"])
    for refused in (report, {"content_changed": ["m.layers.0.mlp.gate.weight"], "layout_changed": []}):
        try:
            override_names(refused, refused is not report)
            raise AssertionError("an expert or router change was accepted")
        except SystemExit:
            pass
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--a", required=True, type=Path, help="the variant's checkpoint directory")
    d.add_argument("--b", required=True, type=Path, help="the reference checkpoint directory")
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--no-hash", action="store_true", help="compare names, dtypes and shapes only")
    o = sub.add_parser("override")
    o.add_argument("--diff", required=True, type=Path, help="a diff report whose content_changed names to fetch")
    o.add_argument("--repo", required=True, help="the variant's BF16 Hugging Face repo (owner/name)")
    o.add_argument("--revision", default="main")
    o.add_argument("--deploy", required=True, type=Path, help="the deployed checkpoint directory the override loads into")
    o.add_argument("--keep", required=True, type=Path, help="the deployment's REAP keep list")
    o.add_argument("--out", required=True, type=Path)
    o.add_argument("--without-experts", action="store_true",
                   help="leave routed-expert changes out (the deployed int4 experts stay); a router change is still refused")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "override":
        names, experts = override_names(json.loads(args.diff.read_text()), args.without_experts)
        source = f"{args.repo}@{args.revision}" + (f" without {len(experts)} routed-expert tensors" if experts else "")
        summary = build_override(names, RemoteCheckpoint(args.repo, args.revision), inventory(args.deploy, False),
                                 args.keep, args.out, source)
        return print(json.dumps(summary, indent=1))
    report = diff(inventory(args.a, not args.no_hash), inventory(args.b, not args.no_hash))
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"counts": report["counts"], "by_category": report["by_category"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
