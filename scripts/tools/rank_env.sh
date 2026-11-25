#!/bin/bash
# torch.distributed.run --no-python target: per-rank Kit cache / home and Warp / OptiX compile caches (two processes
# sharing one stall on its lock files), then exec the command. UWLAB_SHARED_CACHE (downloaded assets, render bank)
# stays shared: fill it with one process before a multi-rank run.
R=${LOCAL_RANK:-0}
export TMPDIR=/tmp/isaaclab_cache/rank$R HOME=/root/rank$R
export WARP_CACHE_PATH=$TMPDIR/warp OPTIX_CACHE_PATH=$TMPDIR/optix
mkdir -p "$TMPDIR" "$HOME/.cache"
[ -f /root/.netrc ] && [ ! -f "$HOME/.netrc" ] && cp /root/.netrc "$HOME/.netrc" && chmod 600 "$HOME/.netrc"
[ -n "${UWLAB_SHARED_CACHE:-}" ] && mkdir -p "$UWLAB_SHARED_CACHE" && ln -sfn "$UWLAB_SHARED_CACHE" "$HOME/.cache/uwlab"
# test only: all ranks on GPU 0 (a one-GPU allocation), keeping their own caches
[ -n "${UWLAB_RANKS_SHARE_GPU:-}" ] && export LOCAL_RANK=0
exec "$@"
