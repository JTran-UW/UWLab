# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Resolve remote (HTTP/HTTPS) USD asset references to local files so the stage
can be opened in Blender, which does not fetch remote references.

Usage: run with uwlab.sh so Isaac Sim is launched (pxr is available there).

    ./uwlab.sh -p scripts/tools/usd_resolve_assets_for_blender.py -i env.usd -o env_blender.usd --headless

With custom assets dir:

    ./uwlab.sh -p scripts/tools/usd_resolve_assets_for_blender.py -i env.usd -o env_blender.usd -a ./env_assets --headless

Remote refs are downloaded into the assets dir and reference paths in the
output USD are rewritten to relative paths under that dir.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import urllib.request

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Resolve remote USD references to local paths for Blender import."
)
parser.add_argument("-i", "--input", required=True, help="Input USD file (e.g. env.usd)")
parser.add_argument("-o", "--output", required=True, help="Output USD file (e.g. env_blender.usd)")
parser.add_argument(
    "-a",
    "--assets-dir",
    default=None,
    help="Directory for downloaded assets (default: <output_dir>/<output_stem>_assets)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pxr import Sdf


def _normalize_http_url(path: str) -> str | None:
    """Return a valid URL if path looks like http(s), else None. Fixes https:/ -> https://."""
    s = path.strip()
    if s.startswith("https:/") and not s.startswith("https://"):
        return "https://" + s[7:]
    if s.startswith("http:/") and not s.startswith("http://"):
        return "http://" + s[6:]
    if s.startswith("https://") or s.startswith("http://"):
        return s
    return None


def _url_to_local_path(url: str, assets_dir: str) -> str:
    """Derive a local path from a URL, preserving path structure under assets_dir."""
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        path = (p.path or "").strip().lstrip("/")
        if not path:
            path = os.path.basename(url) or "asset.usd"
        local = os.path.join(assets_dir, path)
        return local
    except Exception:
        return os.path.join(assets_dir, os.path.basename(url) or "asset.usd")


def _download_asset(url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "UWLab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(local_path, "wb") as f:
        f.write(data)


def _resolve_layer_refs(
    layer, layer_file_path: str, assets_dir: str, refs_done: set
) -> None:
    """Recursively resolve HTTP refs in a layer and any layers it references."""
    try:
        refs = layer.GetExternalReferences()
    except Exception:
        refs = set()
    layer_dir = os.path.dirname(os.path.abspath(layer_file_path))
    for ref in refs:
        if ref in refs_done:
            continue
        url = _normalize_http_url(ref)
        if not url:
            continue
        refs_done.add(ref)
        local_path = _url_to_local_path(url, assets_dir)
        if not os.path.isfile(local_path):
            try:
                _download_asset(url, local_path)
            except Exception as e:
                print(f"Warning: could not download {url}: {e}")
                continue
        try:
            rel = os.path.relpath(os.path.abspath(local_path), layer_dir)
            if rel.startswith(".."):
                rel = os.path.abspath(local_path)
        except ValueError:
            rel = os.path.abspath(local_path)
        layer.UpdateExternalReference(ref, rel)
        try:
            sub = Sdf.Layer.FindOrOpen(local_path)
            if sub and sub.identifier != layer.identifier:
                _resolve_layer_refs(sub, local_path, assets_dir, refs_done)
                sub.Save()
        except Exception:
            pass


def main():
    if not os.path.isfile(args_cli.input):
        raise FileNotFoundError(args_cli.input)
    out_dir = os.path.dirname(os.path.abspath(args_cli.output))
    out_name = os.path.splitext(os.path.basename(args_cli.output))[0]
    assets_dir = args_cli.assets_dir or os.path.join(out_dir, f"{out_name}_assets")
    assets_dir = os.path.abspath(assets_dir)

    input_path = os.path.abspath(args_cli.input)
    layer = Sdf.Layer.FindOrOpen(input_path)
    if layer is None:
        raise RuntimeError(f"Could not open layer: {args_cli.input}")
    refs_done = set()
    _resolve_layer_refs(layer, input_path, assets_dir, refs_done)
    os.makedirs(out_dir, exist_ok=True)
    layer.Export(args_cli.output)
    print(f"Wrote {args_cli.output} with assets under {assets_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
