#!/bin/bash
# Build the halt95 Merlin engine (Flash-Next v2.5.2 = public vLLM v0.30.0 + 98 commits, github.com/halt95/
# qwen38-flash-next-3090s) as one cp312 wheel for sm89 (Kaggle 4x L4, driver CUDA 13.0), plus an offline wheelhouse.
#   source: the v0.30.0 commit + the v2.5.2 release bundle, commit and tree hash checked (upstream/PIN-v2.5), then
#           sm89_qsa_fp8.patch (the fork gates its FP8 QSA KV reader to SM86; the patch admits SM89)
#   compiled ops: the fork's two own extensions (_C_stable_libtorch: top-k kernels; _moe_C_stable_libtorch: Marlin MoE
#           split-K switch) built here for TORCH_CUDA_ARCH_LIST=$ARCH with the CUDA 13.0 nvcc wheel; every other build
#           product is the stock vllm 0.30.0 wheel's, as in the fork's own build-v2.5.sh
#   runtime: the fork's 196 pins (torch 2.13.0 + CUDA 13.0, triton 3.7.1, flashinfer 0.6.18.post1); all have cp312
#           manylinux_2_34 wheels
# Outputs under $W: wheelhouse/ (the 196 pins as cp312 wheels + the vLLM wheel; the env is installed from it offline,
# as Kaggle will), dist/vllm-0.30.0+merlin252.sm89-cp38-abi3-manylinux_2_34_x86_64.whl, logs/, and the new conda env
# $ENV.
set -euo pipefail
W=${W:-$PWD}  # holds this script, sm89_qsa_fp8.patch, requirements-pinned-v2.5.txt and build-artifacts-v2.5.list
ENV=${ENV:-$W/env}
ARCH=${ARCH:-8.9}
LOCAL_VERSION=0.30.0+merlin252.sm89
BASE_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607
BUNDLE_URL=https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz
BUNDLE_SHA256=ab70e1a39768d4ae016f7f9f0ae99165c20c0e0136579f557084218cef8bd685  # of the uncompressed bundle
BUNDLE_REF=refs/heads/v2.5.2-public
COMMIT=de6e2c2fe9e02a9f70a5ff6f313aa45abc43b84c
TREE=05e6d1a583e102b32a133ce34519865cacf32c45
STOCK_URL=https://files.pythonhosted.org/packages/1a/25/f5087a8f382e197a73672768b389d020b9d92e38371c3b0fc6dd883bbb34/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl
STOCK_SHA256=ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4
OWN="_C_stable_libtorch _moe_C_stable_libtorch"
UV=${UV:-uv}

TMP=${TMPDIR:-/tmp}/h95
mkdir -p "$W"/{logs,dist,wheelhouse} "$TMP"
export UV_CACHE_DIR=$TMP/uv PIP_CACHE_DIR=$TMP/pip
cd "$W"
echo "== $(date -Is) host $(hostname), arch $ARCH"

# 1. python env: a new conda env installed offline from the wheelhouse of the fork's pins (cp312 resolves the same 196
#    versions), plus build-only tools from PyPI
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" python=3.12 > logs/conda.log 2>&1
PY=$ENV/bin/python
"$PY" -m pip download -q --no-deps --only-binary :all: -d wheelhouse -r requirements-pinned-v2.5.txt
"$UV" pip install -q --python "$PY" --no-index --find-links wheelhouse --no-deps -r requirements-pinned-v2.5.txt
"$UV" pip install -q --python "$PY" "cmake>=3.26,<4" pytest
"$UV" pip install -q --python "$PY" "nvidia-cuda-cuobjdump==13.0.*" || echo "note: no cuobjdump wheel; SASS arch not listed"
"$PY" -c "import sys, torch; assert sys.version_info[:2] == (3, 12) and torch.__version__.split('+')[0] == '2.13.0' and torch.version.cuda == '13.0', (sys.version, torch.__version__, torch.version.cuda)"
CU=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13
[ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
for so in "$CU"/lib/lib*.so.[0-9]*; do base="${so%%.so.*}.so"; [ -e "$base" ] || ln -s "$(basename "$so")" "$base"; done
NVCC=${NVCC:-$CU/bin/nvcc}
echo "python $("$PY" --version 2>&1), torch $("$PY" -c 'import torch; print(torch.__version__)'), $("$NVCC" --version | grep release)"

# 2. source: v0.30.0 + the release bundle, hashes checked, then the sm89 patch
SRC=$W/src
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" fetch -q --depth 1 https://github.com/vllm-project/vllm.git "$BASE_COMMIT"
  curl -fsSL -o "$TMP/bundle.gz" "$BUNDLE_URL" && gunzip -c "$TMP/bundle.gz" > "$W/v2.5.2.bundle"
  [ "$(sha256sum "$W/v2.5.2.bundle" | cut -d' ' -f1)" = "$BUNDLE_SHA256" ] || { echo "bundle sha256 mismatch"; exit 1; }
  git -C "$SRC" fetch -q "$W/v2.5.2.bundle" "+$BUNDLE_REF:refs/heads/v2.5.2"
  git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$COMMIT"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] && [ "$(git -C "$SRC" rev-parse 'HEAD^{tree}')" = "$TREE" ] \
  || { echo "source is not v2.5.2 ($COMMIT / tree $TREE)"; exit 1; }
git -C "$SRC" diff --quiet || git -C "$SRC" checkout -q -- .
git -C "$SRC" apply "$W/sm89_qsa_fp8.patch"
echo "source: v2.5.2 $COMMIT, tree $TREE verified; applied sm89_qsa_fp8.patch"

# 3. stock build products (everything in build-artifacts.list except the two own extensions)
STOCK=$TMP/$(basename "$STOCK_URL")
[ -f "$STOCK" ] || curl -fsSL -o "$STOCK" "$STOCK_URL"
[ "$(sha256sum "$STOCK" | cut -d' ' -f1)" = "$STOCK_SHA256" ] || { echo "stock wheel sha256 mismatch"; exit 1; }
"$PY" - "$STOCK" "$SRC" build-artifacts-v2.5.list "$OWN" <<'EOF'
import os, sys, zipfile
whl, src, listing, own = sys.argv[1:5]
own = {f"vllm/{name}.abi3.so" for name in own.split()}
want = [line.strip().rstrip("/") for line in open(listing) if line.strip() and line.strip() not in own]
found, count = set(), 0
with zipfile.ZipFile(whl) as archive:
    for info in archive.infolist():
        hit = [w for w in want if info.filename == w or info.filename.startswith(w + "/")]
        if hit and not info.filename.endswith("/"):
            path = archive.extract(info, src)
            count += 1
            found.update(hit)
            if (mode := (info.external_attr >> 16) & 0o777):
                os.chmod(path, mode)
missing = sorted(set(want) - found)
assert not missing, f"stock wheel lacks {missing}"
print(f"stock build products: {count} files")
EOF

# 4. the fork's own extensions for $ARCH (the fork ships them for 8.6 only)
BD=$TMP/build
env PATH="$(dirname "$NVCC"):$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=6 NVCC_THREADS=1 \
  cmake -S "$SRC" -B "$BD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="$PY" -DCMAKE_CUDA_COMPILER="$NVCC" > logs/cmake_configure.log 2>&1 \
  || { tail -40 logs/cmake_configure.log; exit 1; }
start=$SECONDS
for target in $OWN; do
  env PATH="$(dirname "$NVCC"):$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" cmake --build "$BD" --target "$target" -j 6 \
    > "logs/build_$target.log" 2>&1 || { grep -m 20 -E "error|Error" "logs/build_$target.log"; exit 1; }
  cp "$(find "$BD" -name "$target.abi3.so" | head -1)" "$SRC/vllm/$target.abi3.so"
  echo "built $target sha256 $(sha256sum "$SRC/vllm/$target.abi3.so" | cut -d' ' -f1)"
  if [ -x "$CU/bin/cuobjdump" ]; then
    echo "  SASS: $("$CU/bin/cuobjdump" --list-elf "$SRC/vllm/$target.abi3.so" | grep -o 'sm_[0-9]*' | sort | uniq -c | tr '\n' ' ')"
  fi
done
echo "extensions built in $((SECONDS - start)) s"

# 5. the wheel: the stock wheel's metadata and entry points around the fork's vllm/ (Python + the products above) and
#    its two top-level helper packages (the PLE offload process bootstrap and its fault hook); the platform tag says
#    manylinux_2_34 because the two extensions were built against this host's glibc 2.34 (Kaggle has 2.35)
WH=$TMP/wheel && rm -rf "$WH" && mkdir -p "$WH"
"$PY" -m wheel unpack -d "$WH" "$STOCK" > /dev/null
U=$WH/vllm-0.30.0
rm -rf "$U/vllm"
(cd "$SRC" && tar --exclude=__pycache__ -cf - vllm vllm_ple_offload_bootstrap vllm_ple_test_fault) | tar -xf - -C "$U"
mv "$U/vllm-0.30.0.dist-info" "$U/vllm-$LOCAL_VERSION.dist-info"
sed -i "s/^Version: 0.30.0$/Version: $LOCAL_VERSION/" "$U/vllm-$LOCAL_VERSION.dist-info/METADATA"
printf 'vllm_ple_offload_bootstrap\nvllm_ple_test_fault\n' >> "$U/vllm-$LOCAL_VERSION.dist-info/top_level.txt"
rm -f dist/vllm-*.whl
"$PY" -m wheel pack -d dist "$U" > /dev/null
"$PY" -m wheel tags --remove --platform-tag manylinux_2_34_x86_64 dist/vllm-*.whl > /dev/null
WHEEL=$(ls dist/vllm-*.whl)
echo "wheel: $WHEEL $(du -h "$WHEEL" | cut -f1) sha256 $(sha256sum "$WHEEL" | cut -d' ' -f1)"
cp "$WHEEL" wheelhouse/

# 6. install and check without a GPU: imports, the libraries the extensions need, the fork's CPU tests next to the patch
"$UV" pip install -q --python "$PY" --no-index --no-deps --reinstall "$WHEEL"
(cd / && "$PY" -c "import vllm, vllm_ple_offload_bootstrap, vllm_ple_test_fault; from importlib.metadata import version; print('import ok', vllm.__version__, version('vllm'), vllm.__file__)")
if (cd / && "$PY" -c "import vllm.models.qwen4_exp.nvidia.model, vllm.models.qwen4_exp.nvidia.mtp") > logs/import_models.log 2>&1; then
  echo "model modules import ok"
else
  echo "model module import failed on this CPU node: $(tail -1 logs/import_models.log)"
fi
for target in $OWN; do echo "$target needs: $(readelf -d "$SRC/vllm/$target.abi3.so" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' ')"; done
(cd "$SRC" && timeout 1200 "$PY" -m pytest -q -p no:cacheprovider -m "not gpu" \
  tests/models/qwen4_exp/test_qsa_backend_capability.py tests/v1/core/test_side_cache_layout.py \
  tests/config/test_recoverssm_pp_config.py tests/v1/test_e2_ple_mode.py > "$W/logs/pytest_cpu.log" 2>&1) || true
echo "fork CPU tests: $(tail -1 logs/pytest_cpu.log)"
echo "wheelhouse: $(ls wheelhouse | wc -l) files, $(du -sh wheelhouse | cut -f1)"
echo "BUILD_DONE $(date -Is)"
