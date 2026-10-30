#!/bin/bash
# Run one python script from this checkout inside an Isaac Sim 6.1 / Isaac Lab 3 SIF (e.g. uwlab_isaaclab31_cu130.sif,
# Newton 1.5.2, needed by the Real2Sim tasks), with a writable overlay at <repo>/.sif_overlay. The checkout is bound at
# /workspace/uwlab and at its own path; the source/ packages come from the checkout via PYTHONPATH.
# usage: SIF=<image.sif> bash scripts/tools/run_in_sif.sh --setup            # once: pinned rsl-rl-lib + rsl-rl-flashsac + imageio-ffmpeg into the overlay
#        SIF=<image.sif> bash scripts/tools/run_in_sif.sh <script> <args...>  # e.g. scripts/reinforcement_learning/flashsac/eval_video.py ...
set -uo pipefail
E=$(cd "$(dirname "$0")/../.." && pwd); SIF=${SIF:-$HOME/sif/uwlab_isaaclab31_cu130.sif}
R=$E/.sif_overlay; mkdir -p "$R/cache" "$R/home" "$R/overlay/upper" "$R/overlay/work"
[ -f "$R/home/.netrc" ] || { cp "$HOME/.netrc" "$R/home/.netrc" 2>/dev/null && chmod 600 "$R/home/.netrc"; }
W=/workspace/uwlab
PYPATH=$W/source/uwlab:$W/source/uwlab_assets:$W/source/uwlab_rl:$W/source/uwlab_tasks
if [ "${1:-}" = "--setup" ]; then
    # same pins as source/uwlab_rl/setup.py (archives: no git needed inside the image)
    S=$E/source/uwlab_rl/setup.py
    RR=$(grep -oP '^RSL_RL_REPO = "https://github.com/\K[^"]+(?=\.git")' $S); RC=$(grep -oP '^RSL_RL_COMMIT = "\K[0-9a-f]+' $S)
    FR=$(grep -oP '^RSL_RL_FLASHSAC_REPO = "https://github.com/\K[^"]+(?=\.git")' $S); FC=$(grep -oP '^RSL_RL_FLASHSAC_COMMIT = "\K[0-9a-f]+' $S)
    echo "[run_in_sif] installing rsl-rl-lib $RR@$RC, rsl-rl-flashsac $FR@$FC, imageio-ffmpeg into $R/overlay"
    exec apptainer exec --containall --overlay "$R/overlay" --home "$R/home":/root --bind "$R/cache":/tmp/isaaclab_cache:rw \
        --env TMPDIR=/tmp/isaaclab_cache --env PIP_CACHE_DIR=/tmp/isaaclab_cache/pip "$SIF" /opt/venv/bin/python -m pip install --no-deps \
        "https://github.com/$RR/archive/$RC.tar.gz" "https://github.com/$FR/archive/$FC.tar.gz" imageio-ffmpeg
fi
SCRIPT=${1:?script path relative to the repo}; shift
# apptainer --nv does not pass NVIDIA's NGX library (RTX DLSS/denoising) on every host's nvliblist
NGX=""
for f in /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.*.*; do
    [ -e "$f" ] && NGX="$NGX --bind $f:/.singularity.d/libs/$(basename $f)"
done
ENVS=""; [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ENVS="--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for v in $(compgen -e | grep -E "^(UWLAB|FLASHSAC|WANDB)_"); do ENVS="$ENVS --env $v=${!v}"; done
exec apptainer exec --nv --containall $ENVS --overlay "$R/overlay" $NGX \
    --bind "$E":$W:rw --bind "$E":"$E":rw --bind "$R/cache":/tmp/isaaclab_cache:rw --bind /dev/shm:/hostshm \
    --home "$R/home":/root --pwd $W \
    --env TMPDIR=/tmp/isaaclab_cache --env UWLAB_PATH=$W --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONPATH="$PYPATH" \
    "$SIF" /opt/venv/bin/python -u $W/$SCRIPT "$@"
