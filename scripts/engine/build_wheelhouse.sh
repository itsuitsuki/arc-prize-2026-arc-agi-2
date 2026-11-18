#!/bin/bash
# Assemble the offline wheelhouse of halt95's Merlin engine (the "merlin" engine: vLLM 0.30.0 + the v2.5.2 patch series,
# built for sm89) for the Kaggle notebook (Python 3.12, x86_64, glibc 2.35), then install it the way the notebook does.
#   wheels: the 196 cp312 wheels of halt95's release/v2.5/requirements-pinned.txt and our
#           vllm-0.30.0+merlin252.sm89 wheel, all from the wheel build (WHEELS, its build_wheel.sh)
#   flashinfer-aot-sm89-<module>.so: FlashInfer's sampling and norm JIT modules compiled for sm89 here, so the server
#           does not compile them at its first requests; the notebook copies them to <venv>/flashinfer/data/aot/<module>/,
#           where FlashInfer loads a module without building it
#   manifest.json: every file's size and sha256 (wheels also name and version), the install check and the timings
# Install check: a fresh venv without pip from a Python 3.12, `pip --python <venv> install --no-index --require-hashes
# --no-deps` from the wheel files (halt95's pins are a Python 3.13 freeze, and on 3.12 mistral-common 1.11.7 asks for
# numpy<2.4 against the pinned 2.4.6, which pip's resolver refuses), `pip check` for the record, imports, and the AOT
# modules loading with FlashInfer's JIT disabled. CPU only: the compile needs nvcc (the wheels' CUDA 13.0) and a
# libcuda stub to link.
set -euo pipefail
WHEELS=${WHEELS:?the wheelhouse/ folder that build_wheel.sh writes}
OUT=${OUT:?output folder}
PY=${PY:?a Python 3.12 with pip}  # Python 3.12 with pip; the venv sees none of its packages
STUBS=${STUBS:?the stubs/ folder of a CUDA toolkit}             # libcuda.so to link against (no driver on a CPU node)
WH=$OUT/wheels
TMP=${TMPDIR:-/tmp}/merlinwh
VENV=$TMP/venv
mkdir -p "$WH" "$OUT/logs" "$TMP"
cd /  # nothing on the path may shadow an installed package
echo "== $(date -Is) host $(hostname), $(nproc) cores"

# 1. the wheels (hard links into the dataset folder)
for wheel in "$WHEELS"/*.whl; do ln -f "$wheel" "$WH/" 2>/dev/null || cp -f "$wheel" "$WH/"; done
echo "wheels: $(ls "$WH"/*.whl | wc -l)"

# 2. a fresh venv and the notebook's install command
start=$SECONDS
"$PY" -m venv --without-pip --clear "$VENV"
"$PY" - "$WH" "$TMP/requirements.txt" <<'EOF'
import hashlib, sys
from pathlib import Path
wheels, out = Path(sys.argv[1]), Path(sys.argv[2])
out.write_text("".join(f"{w.as_uri()} --hash=sha256:{hashlib.sha256(w.read_bytes()).hexdigest()}\n" for w in sorted(wheels.glob("*.whl"))))
EOF
hashed=$((SECONDS - start)); start=$SECONDS
"$PY" -m pip --python "$VENV/bin/python" install --no-index --find-links "$WH" --only-binary=:all: --require-hashes \
  --no-compile --no-deps -q -r "$TMP/requirements.txt" > "$OUT/logs/pip_install.log" 2>&1 || { tail -30 "$OUT/logs/pip_install.log"; exit 1; }
installed=$((SECONDS - start))
"$PY" -m pip --python "$VENV/bin/python" check > "$OUT/logs/pip_check.log" 2>&1 || true
echo "pip check: $(cat "$OUT/logs/pip_check.log" | tr '\n' ';')"
echo "venv install: ${installed} s (hashing ${hashed} s), $(du -sh "$VENV" | cut -f1), host pip $("$PY" -m pip --version | cut -d' ' -f2)"
"$VENV/bin/python" -m pip --version > /dev/null 2>&1 && echo "note: the venv has its own pip" || true
"$VENV/bin/python" -c "
import json, sys, torch, flashinfer, vllm, vllm_ple_offload_bootstrap, transformers, openai, httpx, numpy
from importlib.metadata import version
assert sys.version_info[:2] == (3, 12) and torch.version.cuda == '13.0', (sys.version, torch.version.cuda)
print('IMPORT_OK', json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'torch.version.cuda': torch.version.cuda,
      'vllm': version('vllm'), 'flashinfer': flashinfer.__version__, 'transformers': transformers.__version__, 'openai': openai.__version__,
      'httpx': httpx.__version__, 'numpy': numpy.__version__}))" 2>&1 | grep -v Warning | tee "$OUT/logs/import_check.log"

# 3. FlashInfer's sampling and norm modules for sm89, in the serve script's toolchain (the wheels' CUDA 13.0 nvcc; the
#    wheel ships only libcudart.so.13, so libcudart.so is linked in)
PURELIB=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
CU=$PURELIB/nvidia/cu13
mkdir -p "$TMP/cudalink" && ln -sf "$CU/lib/libcudart.so.13" "$TMP/cudalink/libcudart.so"
export CUDA_HOME=$CU PATH="$CU/bin:$VENV/bin:$PATH" LIBRARY_PATH="$TMP/cudalink:$CU/lib:$STUBS"
export FLASHINFER_CUDA_ARCH_LIST=8.9 FLASHINFER_WORKSPACE_BASE=$TMP/flashinfer-build MAX_JOBS=$(nproc)
echo "nvcc: $(nvcc --version | grep release)"
"$VENV/bin/python" - "$WH" <<'EOF' 2>&1 | tee "$OUT/logs/flashinfer_aot.log"
import json, shutil, sys, time
from pathlib import Path
from flashinfer.jit.norm import gen_norm_module
from flashinfer.jit.sampling import gen_sampling_module
seconds = {}
for gen in (gen_sampling_module, gen_norm_module):
    spec, start = gen(), time.time()
    spec.build(verbose=False)
    seconds[spec.name] = round(time.time() - start, 1)
    shutil.copy(spec.jit_library_path, Path(sys.argv[1]) / f"flashinfer-aot-sm89-{spec.name}.so")
print("AOT_BUILT", json.dumps(seconds))
EOF

# 4. the AOT modules load from <venv>/flashinfer/data/aot with JIT disabled (the stub stands in for libcuda.so.1 here)
for so in "$WH"/flashinfer-aot-sm89-*.so; do
  name=$(basename "$so" .so); name=${name#flashinfer-aot-sm89-}
  mkdir -p "$PURELIB/flashinfer/data/aot/$name" && cp "$so" "$PURELIB/flashinfer/data/aot/$name/$name.so"
done
mkdir -p "$TMP/driverstub" && ln -sf "$STUBS/libcuda.so" "$TMP/driverstub/libcuda.so.1"
LD_LIBRARY_PATH=$TMP/driverstub FLASHINFER_DISABLE_JIT=1 FLASHINFER_WORKSPACE_BASE=$TMP/flashinfer-empty \
  "$VENV/bin/python" -c "
from flashinfer.norm import get_norm_module
from flashinfer.sampling import get_sampling_module
get_sampling_module(); get_norm_module(); print('AOT_LOAD_OK')" 2>&1 | grep -v Warning | tee "$OUT/logs/aot_load.log"
readelf -d "$WH/flashinfer-aot-sm89-sampling.so" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' '; echo

# 5. the manifest
"$PY" - "$WH" "$OUT/logs" "$installed" <<'EOF'
import hashlib, json, re, sys
from pathlib import Path
from packaging.utils import parse_wheel_filename
wheels, logs = Path(sys.argv[1]), Path(sys.argv[2])
def record(path):
    return {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
files = {}
for path in sorted(wheels.glob("*.whl")):
    name, version, _, _ = parse_wheel_filename(path.name)
    files[path.name] = {"name": str(name), "version": str(version), **record(path)}
imports = re.search(r"IMPORT_OK (.*)", (logs / "import_check.log").read_text())
built = re.search(r"AOT_BUILT (.*)", (logs / "flashinfer_aot.log").read_text())
manifest = {"status": "WHEELHOUSE_INSTALLED_OFFLINE" if imports and "AOT_LOAD_OK" in (logs / "aot_load.log").read_text() else "CHECK_FAILED",
            "python": "3.12", "platform": "x86_64 manylinux (Kaggle: glibc 2.35)", "files": files,
            "flashinfer_aot": {p.name: {"module": p.stem.removeprefix("flashinfer-aot-sm89-"), "arch": "sm_89", **record(p)}
                               for p in sorted(wheels.glob("flashinfer-aot-sm89-*.so"))},
            "install_check": json.loads(imports.group(1)) if imports else None, "install_seconds": int(sys.argv[3]),
            "aot_compile_seconds": json.loads(built.group(1)) if built else None, "gpu_tested": False}
(wheels / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest["status"], len(files), "wheels", round(sum(r["size_bytes"] for r in files.values()) / 2**30, 2), "GiB")
EOF
echo "dataset folder: $(ls "$WH" | wc -l) files, $(du -sh "$WH" | cut -f1)"
echo "BUILD_DONE $(date -Is)"
