#!/bin/bash
# Build the Python packages of the OptiX renderer backend into <repo>/_optix_deps/<flavor>. Put that directory FIRST on
# PYTHONPATH (it carries Warp 1.19, which replaces the image's Warp 1.16 for the whole process).
#   stock : PyPI warp-lang + warp-optix + pyoptix, built against OptiX SDK 9.1 -> NVIDIA driver R590 or newer
#   r570  : the same, with pyoptix and the warp-optix bridge recompiled against the OptiX 9.0 headers
#           -> driver R570 or newer (R580 nodes fail the stock wheels with "ImportError: Unsupported ABI version")
# usage: bash scripts/tools/build_optix_deps.sh [stock|r570] [python]     (python must be 3.12; r570 needs g++, ~5 min)
set -euo pipefail
FLAVOR=${1:-r570}; PY=${2:-python3}
case "$FLAVOR" in stock|r570) ;; *) echo "usage: $0 [stock|r570] [python]"; exit 2 ;; esac
E=$(cd "$(dirname "$0")/../.." && pwd); OUT=${OPTIX_DEPS_OUT:-$E/_optix_deps/$FLAVOR}
WARP=1.19.0; WARP_OPTIX=0.3.0; PYOPTIX=9.1.1
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "warp-optix needs Python 3.12"'
rm -rf "$OUT"; mkdir -p "$OUT"
"$PY" -m pip install -q --target "$OUT" --no-deps "warp-lang==$WARP" "warp-optix==$WARP_OPTIX" "pyoptix==$PYOPTIX"
if [ "$FLAVOR" = stock ]; then echo "[optix_deps] wrote $OUT (OptiX 9.1, driver R590+)"; exit 0; fi

B=$(mktemp -d "${TMPDIR:-/tmp}/optix_deps.XXXXXX"); trap 'rm -rf "$B"' EXIT
fetch() {  # url sha256 file
    curl -sSfL -o "$B/$3" "$1"
    echo "$2  $B/$3" | sha256sum -c --quiet
}
fetch https://files.pythonhosted.org/packages/db/90/f61ceaf0d85592296f611bf58fc469e7118a5b28be4bc5fcb4345427fe98/warp_optix-0.3.0.tar.gz \
    9695d5c81a88eb4d6297cb2af654658ce10cac504c11d5b4316b6059e836a348 warp_optix.tar.gz
fetch https://files.pythonhosted.org/packages/62/34/19c969b2c557c6244a3c83f74b61ad2d5aaf798a29926908a6252f9c9649/pyoptix-9.1.1.tar.gz \
    7255570544a5fdf2885998231172edc1978bce1fe883f86e9940a0ff2192f791 pyoptix.tar.gz
# NVIDIA/optix-dev tag v9.0.0 (OPTIX_ABI_VERSION 105)
fetch https://github.com/NVIDIA/optix-dev/archive/fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd.tar.gz \
    d5239da7ad098ae24afc6651d0cd3020ab0346a575ff4935a60e5494fd0bbe28 optix_dev.tar.gz
for a in warp_optix pyoptix optix_dev; do mkdir "$B/$a"; tar -xzf "$B/$a.tar.gz" -C "$B/$a" --strip-components=1; done
grep -q "define OPTIX_ABI_VERSION 105" "$B/optix_dev/include/optix_function_table.h"
# build-only headers: pybind11 and cuda.h
"$PY" -m pip install -q --target "$B/tools" --no-deps "pybind11>=3.0,<4" "nvidia-cuda-runtime-cu12==12.9.79"
INC="-I$B/tools/pybind11/include -I$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])') -I$B/optix_dev/include -I$B/tools/nvidia/cuda_runtime/include"
SO=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
CXX="${CXX:-g++} -O2 -fPIC -shared -std=c++17 -fvisibility=hidden"

# pyoptix: only its cudaFree(0) "init the CUDA runtime" call needs libcudart; Warp owns the CUDA context here.
sed -e 's|#include <cuda_runtime.h>|#include <cuda.h>|' -e 's|^    cudaFree(0);.*|    // CUDA is initialised by Warp|' \
    "$B/pyoptix/src/main.cpp" > "$B/pyoptix_main.cpp"
! grep -q "cudaFree" "$B/pyoptix_main.cpp"
echo "[optix_deps] compiling pyoptix against OptiX 9.0 (a few minutes)"
$CXX -DVERSION_INFO=$PYOPTIX "$B/pyoptix_main.cpp" $INC -o "$OUT/optix/_optix$SO" -ldl
# warp-optix bridge: drop its "OptiX 9.1.0 only" assertion
sed 's|static_assert(OPTIX_VERSION == 90100, .*|static_assert(OPTIX_VERSION == 90000, "this warp-optix bridge build targets OptiX 9.0.0");|' \
    "$B/warp_optix/src/extension.cpp" > "$B/warp_optix/src/extension_90.cpp"
grep -q "OPTIX_VERSION == 90000" "$B/warp_optix/src/extension_90.cpp"
echo "[optix_deps] compiling the warp-optix bridge against OptiX 9.0"
$CXX -fvisibility-inlines-hidden "$B/warp_optix/src/extension_90.cpp" -I"$B/warp_optix/src" $INC -o "$OUT/warp_optix/_ext$SO" -ldl
# device headers that Warp compiles kernels against must match the host ABI
rm -rf "$OUT/warp_optix/include"; cp -r "$B/optix_dev/include" "$OUT/warp_optix/include"
cp "$B/optix_dev/LICENSE.txt" "$OUT/warp_optix/include/OptiX_LICENSE.txt"
"$PY" - "$OUT/warp_optix" <<'EOF'
import pathlib, sys

root = pathlib.Path(sys.argv[1])
# accept the SDK both extensions were rebuilt against
path = root / "_src/backend.py"
text = path.read_text()
old = "if upstream_sdk != extension_sdk or extension_sdk != (9, 1, 0):"
assert text.count(old) == 1, "warp-optix backend.py changed; update this patch"
path.write_text(text.replace(old, "if upstream_sdk != extension_sdk or extension_sdk not in ((9, 0, 0), (9, 1, 0)):"))
# OptiX 9.0 has no cooperative vectors under NVRTC: compile that section of the device builtins out
path = root / "_native/include/warp_optix_builtins.h"
text = path.read_text()
start = text.index("// ---------------------------------------------------------------------------\n// Cooperative vectors.")
end = text.rindex("}  // namespace wp")
guard = "#if defined(OPTIX_INCLUDE_COOPERATIVE_VECTOR) && OPTIX_INCLUDE_COOPERATIVE_VECTOR\n"
path.write_text(text[:start] + guard + text[start:end] + "#endif  // OPTIX_INCLUDE_COOPERATIVE_VECTOR\n\n" + text[end:])
EOF
PYTHONPATH="$OUT" "$PY" -c 'import optix, warp_optix; print("[optix_deps] pyoptix SDK", optix.version(), "warp-optix", warp_optix.__version__)' \
    || echo "[optix_deps] import check failed on this machine (needs an NVIDIA driver >= R570); the build itself completed"
echo "[optix_deps] wrote $OUT (OptiX 9.0, driver R570+)"
