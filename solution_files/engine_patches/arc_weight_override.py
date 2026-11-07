import os
import sys


OFFLOADED = ".ple.ple_embedding."  # Merlin's PleOffloadLayer subtree, the only tensors in the PLE offload worker's load stream


def patch_weight_override(module, path: str, class_name: str, keep: str | None = None) -> None:
    from pathlib import Path

    from safetensors import safe_open

    from arc_reap import keep_sha256

    with safe_open(path, framework="pt") as stream:
        names, metadata = set(stream.keys()), stream.metadata() or {}
    expected, actual = metadata.get("reap_keep_sha256"), keep_sha256(Path(keep)) if keep else None
    if expected != actual:
        raise ValueError(f"{path} was trained under REAP keep list sha256 {expected}, the server prunes with {actual}")
    cls = getattr(module, class_name)
    original = cls.load_weights
    modules = {name.rsplit(".", 2)[0] for name in names} | {name.rsplit(".", 1)[0] for name in names}

    def load_weights(self, weights):
        seen, touched, other = set(), [], []
        with safe_open(path, framework="pt") as stream:
            def substituted():
                for name, tensor in weights:
                    if not other and OFFLOADED not in name:
                        other.append(name)
                    if not touched and (name.rsplit(".", 1)[0] in modules or name.rsplit(".", 2)[0] in modules):
                        touched.append(name)
                    if name in names:
                        new = stream.get_tensor(name)
                        if new.shape != tensor.shape or new.dtype != tensor.dtype:
                            raise ValueError(f"override {name} is {new.dtype} {tuple(new.shape)}, the checkpoint loads "
                                             f"{tensor.dtype} {tuple(tensor.shape)}")
                        seen.add(name)
                        tensor = new
                    yield name, tensor

            result = original(self, substituted())
        required = names if other else {name for name in names if OFFLOADED in name}
        if required - seen and (seen or touched):
            raise ValueError(f"{len(required - seen)} override tensors never reached the loader, e.g. {sorted(required - seen)[:3]}")
        print(f"[weight_override] pid {os.getpid()}: substituted {len(seen)} of {len(names)} tensors from {path}",
              file=sys.stderr, flush=True)
        return result

    cls.load_weights = load_weights

from arc_patches import chain
if os.environ.get("ARC_VLLM_WEIGHT_OVERRIDE"):  # the REAP hook below wraps this loader, so it sees the pruned, renumbered stream
    chain("vllm.models.qwen4_exp.nvidia.model", lambda module, path=os.environ["ARC_VLLM_WEIGHT_OVERRIDE"], keep=os.environ.get("ARC_VLLM_REAP_KEEP"):
          patch_weight_override(module, path, "Qwen4ExpForConditionalGeneration", keep))
