"""Diff two IsaacLab configclasses directly — no intermediate YAML files.

Must be run with Isaac Sim active (same as play.py / train.py).

Usage:
    python scripts/diff_configclasses.py <ClassA> <ClassB> [--no-color]

Examples:
    python scripts/diff_configclasses.py \\
        uwlab_tasks...depth_dagger_cfg.TeacherProprioWithPCCfg \\
        uwlab_tasks...rgb_dagger_cfg.Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg \\
        --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diff two configclasses.")
parser.add_argument("class_a", help="Dotted path to first configclass")
parser.add_argument("class_b", help="Dotted path to second configclass")
parser.add_argument("--no-color", action="store_true", help="Disable color output")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
import enum
import importlib
import re
from typing import Any

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# ── ANSI ──────────────────────────────────────────────────────────────────────

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

USE_COLOR = not args_cli.no_color and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


# ── configclass → plain dict ───────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_dict(v) for v in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    return repr(obj)


def _load_class(dotted: str):
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"[ERROR] Cannot import '{module_path}': {e}", file=sys.stderr)
        sys.exit(1)
    cls = getattr(module, class_name, None)
    if cls is None:
        print(f"[ERROR] '{class_name}' not found in '{module_path}'", file=sys.stderr)
        sys.exit(1)
    if not dataclasses.is_dataclass(cls):
        print(f"[ERROR] '{dotted}' is not a dataclass/configclass", file=sys.stderr)
        sys.exit(1)
    try:
        return class_name, _to_dict(cls())
    except Exception as e:
        print(f"[ERROR] Cannot instantiate '{class_name}': {e}", file=sys.stderr)
        sys.exit(1)


# ── diff helpers (mirrors diff_configs.py) ────────────────────────────────────

_ADDR_RE = re.compile(r" at 0x[0-9a-f]+")


def _norm(v: Any) -> Any:
    return _ADDR_RE.sub("", v) if isinstance(v, str) else v


def _eq(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return _norm(a) == _norm(b)


def _is_term(d: Any) -> bool:
    return isinstance(d, dict) and "func" in d


def _fmt(v: Any, max_len: int = 100) -> str:
    if v is None:
        return "null"
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, full))
            else:
                out[full] = v
    else:
        out[prefix] = obj
    return out


def _print_term_added(key: str, term: Any, pad: str) -> None:
    print(f"{pad}{_c(GREEN, '+')} {_c(BOLD, key)}")
    if isinstance(term, dict):
        for k, v in term.items():
            print(f"{pad}      {_c(DIM, k + ':')} {_c(GREEN, '{...}' if isinstance(v, dict) else _fmt(v))}")
    else:
        print(f"{pad}      {_c(GREEN, _fmt(term))}")


def _print_term_removed(key: str, term: Any, pad: str) -> None:
    print(f"{pad}{_c(RED, '-')} {_c(BOLD, key)}")
    if isinstance(term, dict):
        for k, v in term.items():
            print(f"{pad}      {_c(DIM, k + ':')} {_c(RED, '{...}' if isinstance(v, dict) else _fmt(v))}")
    else:
        print(f"{pad}      {_c(RED, _fmt(term))}")


def _print_term_changed(key: str, a: Any, b: Any, pad: str) -> None:
    flat_a = _flatten(a) if isinstance(a, dict) else {"": a}
    flat_b = _flatten(b) if isinstance(b, dict) else {"": b}
    all_keys = list(dict.fromkeys(list(flat_a) + list(flat_b)))
    changes = [(k, flat_a.get(k), flat_b.get(k)) for k in all_keys if not _eq(flat_a.get(k), flat_b.get(k))]
    if not changes:
        return
    n = len(changes)
    print(f"{pad}{_c(YELLOW, '~')} {_c(BOLD, key)}  {_c(DIM, f'({n} field{chr(115) if n > 1 else chr(32)} changed)'.strip())}")
    for field, va, vb in changes:
        label = field if field else key
        if va is None:
            print(f"{pad}      {_c(GREEN, f'+ {label}:')} {_c(GREEN, _fmt(vb))}")
        elif vb is None:
            print(f"{pad}      {_c(RED, f'- {label}:')} {_c(RED, _fmt(va))}")
        else:
            print(f"{pad}      {_c(DIM, label + ':')}  {_c(RED, _fmt(va))}  →  {_c(GREEN, _fmt(vb))}")


def _print_section(a: dict, b: dict, indent: int = 0) -> None:
    pad = "  " * indent
    for key in dict.fromkeys(list(a) + list(b)):
        in_a, in_b = key in a, key in b
        av, bv = a.get(key), b.get(key)
        if in_a and in_b and _eq(av, bv):
            continue
        if not in_b:
            _print_term_removed(key, av, pad) if isinstance(av, dict) else print(f"{pad}{_c(RED, '-')} {key}: {_c(RED, _fmt(av))}")
        elif not in_a:
            _print_term_added(key, bv, pad) if isinstance(bv, dict) else print(f"{pad}{_c(GREEN, '+')} {key}: {_c(GREEN, _fmt(bv))}")
        elif _is_term(av) or _is_term(bv):
            _print_term_changed(key, av, bv, pad)
        elif isinstance(av, dict) and isinstance(bv, dict):
            print(f"\n{pad}{_c(CYAN, key)}")
            _print_section(av, bv, indent + 1)
        else:
            print(f"{pad}{_c(YELLOW, '~')} {key}:  {_c(RED, _fmt(av))}  →  {_c(GREEN, _fmt(bv))}")


def _count(a: Any, b: Any) -> tuple[int, int, int]:
    nc = na = nr = 0
    if not isinstance(a, dict) or not isinstance(b, dict):
        return (1, 0, 0) if not _eq(a, b) else (0, 0, 0)
    for k in set(a) | set(b):
        if k not in b:
            nr += 1
        elif k not in a:
            na += 1
        elif not _eq(a[k], b[k]):
            if _is_term(a[k]) or _is_term(b[k]) or not isinstance(a[k], dict):
                nc += 1
            else:
                c2, a2, r2 = _count(a[k], b[k])
                nc += c2; na += a2; nr += r2
    return nc, na, nr


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    name_a, data_a = _load_class(args_cli.class_a)
    name_b, data_b = _load_class(args_cli.class_b)

    nc, na, nr = _count(data_a, data_b)
    total = nc + na + nr

    W = 57
    print(_c(BOLD, f"\n{'─' * W}"))
    print(f"  {_c(DIM, 'A:')} {name_a}")
    print(f"  {_c(DIM, 'B:')} {name_b}")
    print(_c(BOLD, f"{'─' * W}"))
    if total == 0:
        print(_c(DIM, "  (identical)"))
    else:
        print(f"  {_c(YELLOW, f'~ {nc} changed')}   {_c(GREEN, f'+ {na} added')}   {_c(RED, f'- {nr} removed')}")
    print()

    for key in dict.fromkeys(list(data_a) + list(data_b)):
        av, bv = data_a.get(key), data_b.get(key)
        if _eq(av, bv):
            continue
        print(_c(BOLD, _c(CYAN, key)))
        if isinstance(av, dict) and isinstance(bv, dict):
            _print_section(av, bv, indent=1)
        elif av is None:
            _print_term_added(key, bv, "  ")
        elif bv is None:
            _print_term_removed(key, av, "  ")
        else:
            print(f"  {_c(YELLOW, '~')} {_c(RED, _fmt(av))}  →  {_c(GREEN, _fmt(bv))}")
        print()

    print(_c(BOLD, f"{'─' * W}\n"))
    sys.exit(0 if total == 0 else 1)


main()
simulation_app.close()
