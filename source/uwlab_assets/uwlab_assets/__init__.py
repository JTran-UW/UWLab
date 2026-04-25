# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import logging
import os
import toml
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Conveniences to other module directories via relative paths
UWLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
UWLAB_ASSETS_DATA_DIR = os.path.join(UWLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
UWLAB_ASSETS_METADATA = toml.load(os.path.join(UWLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/patrickhaoy/uwlab-assets/resolve/main"


def _parse_hf_url(url: str) -> tuple[str, str, str, str] | None:
    """Parse a HuggingFace resolve URL into (repo_id, repo_type, revision, filename).

    Examples:
        ``https://huggingface.co/datasets/patrickhaoy/uwlab-assets/resolve/main/Props/X/y.usd``
            -> ("patrickhaoy/uwlab-assets", "dataset", "main", "Props/X/y.usd")
        ``https://huggingface.co/patrickhaoy/some-model/resolve/v2.0/file.bin``
            -> ("patrickhaoy/some-model", "model", "v2.0", "file.bin")

    Returns None if the URL doesn't look like a HuggingFace resolve URL.
    """
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = parsed.path.strip("/").split("/")
    try:
        resolve_idx = parts.index("resolve")
    except ValueError:
        return None
    if resolve_idx + 2 > len(parts) - 1:
        return None
    if parts[0] == "datasets":
        repo_type = "dataset"
        repo_parts = parts[1:resolve_idx]
    elif parts[0] == "spaces":
        repo_type = "space"
        repo_parts = parts[1:resolve_idx]
    else:
        repo_type = "model"
        repo_parts = parts[:resolve_idx]
    repo_id = "/".join(repo_parts)
    revision = parts[resolve_idx + 1]
    filename = "/".join(parts[resolve_idx + 2 :])
    return repo_id, repo_type, revision, filename


def _urlretrieve_quiet(url: str, dest: str) -> None:
    """Download *url* to *dest* silently. Used as a fallback for non-HF HTTP URLs."""
    req = urllib.request.urlopen(url)
    chunk_size = 1 << 16  # 64 KiB
    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    req.close()


def resolve_cloud_path(path: str) -> str:
    """Resolve a cloud asset path to a local file, downloading if needed.

    * Local paths (including already-cached files) are returned immediately.
    * HuggingFace HTTPS URLs are routed through ``hf_hub_download``, which uses
      the standard HF Hub cache (~/.cache/huggingface/hub/) with ETag-based
      revision tracking. When the remote file changes, the cache is invalidated
      automatically — no manual cache cleanup needed across hosts.
    * Other HTTPS URLs fall back to a simple urllib download into
      ``~/.cache/uwlab/assets/<relative>`` (one-shot, no staleness check).
    """
    if not path.startswith(("http://", "https://")):
        return path

    hf_parts = _parse_hf_url(path)
    if hf_parts is not None:
        from huggingface_hub import hf_hub_download

        repo_id, repo_type, revision, filename = hf_parts
        return hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filename,
        )

    # Non-HF fallback: simple urllib download to legacy uwlab cache
    parsed = urlparse(path)
    rel = parsed.path.strip("/")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "uwlab", "assets")
    local = os.path.join(cache_dir, rel)
    if os.path.isfile(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = f"{local}.tmp.{os.getpid()}"
    try:
        logger.info(f"Downloading {rel} ...")
        _urlretrieve_quiet(path, tmp)
        os.rename(tmp, local)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return local


# Configure the module-level variables
__version__ = UWLAB_ASSETS_METADATA["package"]["version"]
