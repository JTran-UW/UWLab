"""Export an IsaacLab configclass to a YAML file.

Must be run with Isaac Sim active (same as play.py / train.py).

Usage:
    python scripts/export_config_yaml.py <dotted.module.ClassName> [output.yaml]

Examples:
    python scripts/export_config_yaml.py \\
        uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rgb_dagger_cfg.Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg \\
        my_cfg.yaml
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a configclass to YAML.")
parser.add_argument("class_path", help="Dotted path to the configclass, e.g. my.module.MyCfg")
parser.add_argument("output", nargs="?", help="Output YAML file path (default: <ClassName>.yaml)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
import enum
import importlib
from pathlib import Path

import yaml

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401


def _to_yaml_safe(obj):
    """Recursively convert a configclass instance to a YAML-safe structure."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_yaml_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_yaml_safe(v) for v in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    # Class references, callables, tensors, etc. — store as repr string
    return repr(obj)


def main():
    module_path, class_name = args_cli.class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"[ERROR] Could not import module '{module_path}': {e}", file=sys.stderr)
        sys.exit(1)

    cls = getattr(module, class_name, None)
    if cls is None:
        print(f"[ERROR] Class '{class_name}' not found in module '{module_path}'", file=sys.stderr)
        sys.exit(1)
    if not dataclasses.is_dataclass(cls):
        print(f"[ERROR] '{args_cli.class_path}' is not a dataclass/configclass", file=sys.stderr)
        sys.exit(1)

    try:
        instance = cls()
    except Exception as e:
        print(f"[ERROR] Could not instantiate '{class_name}' with defaults: {e}", file=sys.stderr)
        sys.exit(1)

    data = _to_yaml_safe(instance)
    output_path = args_cli.output or f"{class_name}.yaml"
    Path(output_path).write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))
    print(f"[INFO] Saved {class_name} → {output_path}")


main()
simulation_app.close()
