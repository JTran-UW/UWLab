"""Diff two YAML config files with a hierarchical, term-aware summary.

Terms (dicts with a top-level 'func' key) are treated as atomic units — changes
within a term are shown as a flat field list rather than recursing further.

Usage:
    python scripts/diff_configs.py config_a.yaml config_b.yaml [--no-color]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

USE_COLOR = True


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


# ── value helpers ─────────────────────────────────────────────────────────────

_ADDR_RE = re.compile(r" at 0x[0-9a-f]+")

def _norm(v: Any) -> Any:
    """Normalize values for comparison — strip memory addresses from func reprs."""
    if isinstance(v, str):
        return _ADDR_RE.sub("", v)
    return v

def _fmt(v: Any, max_len: int = 100) -> str:
    if v is None:
        return "null"
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _eq(a: Any, b: Any) -> bool:
    """Deep equality with address-normalized comparison."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return _norm(a) == _norm(b)


# ── term detection ─────────────────────────────────────────────────────────────

def _is_term(d: Any) -> bool:
    """A 'term' is an atomic config unit: a dict with a top-level 'func' key."""
    return isinstance(d, dict) and "func" in d


# ── flatten a term for field-level comparison ──────────────────────────────────

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


# ── per-term printers ─────────────────────────────────────────────────────────

def _print_term_added(key: str, term: Any, pad: str) -> None:
    print(f"{pad}{_c(GREEN, '+')} {_c(BOLD, key)}")
    if isinstance(term, dict):
        for k, v in term.items():
            vs = "{...}" if isinstance(v, dict) else _fmt(v)
            print(f"{pad}      {_c(DIM, k + ':')} {_c(GREEN, vs)}")
    else:
        print(f"{pad}      {_c(GREEN, _fmt(term))}")


def _print_term_removed(key: str, term: Any, pad: str) -> None:
    print(f"{pad}{_c(RED, '-')} {_c(BOLD, key)}")
    if isinstance(term, dict):
        for k, v in term.items():
            vs = "{...}" if isinstance(v, dict) else _fmt(v)
            print(f"{pad}      {_c(DIM, k + ':')} {_c(RED, vs)}")
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
    suffix = _c(DIM, f"({n} field{'s' if n > 1 else ''} changed)")
    print(f"{pad}{_c(YELLOW, '~')} {_c(BOLD, key)}  {suffix}")
    for field, va, vb in changes:
        label = field if field else key
        if va is None:
            print(f"{pad}      {_c(GREEN, f'+ {label}:')} {_c(GREEN, _fmt(vb))}")
        elif vb is None:
            print(f"{pad}      {_c(RED, f'- {label}:')} {_c(RED, _fmt(va))}")
        else:
            print(f"{pad}      {_c(DIM, label + ':')}  {_c(RED, _fmt(va))}  →  {_c(GREEN, _fmt(vb))}")


# ── recursive section differ ───────────────────────────────────────────────────

def _has_changes(a: Any, b: Any) -> bool:
    return not _eq(a, b)


def _print_section(a: dict, b: dict, indent: int = 0) -> bool:
    """
    Recursively diff two dicts. Terms (func-bearing dicts) are atomic.
    Returns True if any output was printed.
    """
    pad = "  " * indent
    all_keys = list(dict.fromkeys(list(a) + list(b)))
    printed_any = False

    for key in all_keys:
        in_a, in_b = key in a, key in b
        av, bv = a.get(key), b.get(key)

        if in_a and in_b and not _has_changes(av, bv):
            continue

        printed_any = True

        if not in_b:
            # Removed
            if isinstance(av, dict):
                _print_term_removed(key, av, pad)
            else:
                print(f"{pad}{_c(RED, '-')} {key}: {_c(RED, _fmt(av))}")

        elif not in_a:
            # Added
            if isinstance(bv, dict):
                _print_term_added(key, bv, pad)
            else:
                print(f"{pad}{_c(GREEN, '+')} {key}: {_c(GREEN, _fmt(bv))}")

        elif _is_term(av) or _is_term(bv):
            # Atomic term — show flat field diff
            _print_term_changed(key, av, bv, pad)

        elif isinstance(av, dict) and isinstance(bv, dict):
            # Structural group — recurse under a header
            print(f"\n{pad}{_c(CYAN, key)}")
            _print_section(av, bv, indent + 1)

        else:
            # Scalar changed
            print(f"{pad}{_c(YELLOW, '~')} {key}:  {_c(RED, _fmt(av))}  →  {_c(GREEN, _fmt(bv))}")

    return printed_any


# ── change counter (for summary) ───────────────────────────────────────────────

def _count(a: Any, b: Any) -> tuple[int, int, int]:
    """Return (changed, added, removed) leaf-change counts."""
    nc = na = nr = 0
    if not isinstance(a, dict) or not isinstance(b, dict):
        return (1, 0, 0) if _has_changes(a, b) else (0, 0, 0)
    for k in set(a) | set(b):
        if k not in b:
            nr += 1
        elif k not in a:
            na += 1
        elif _has_changes(a[k], b[k]):
            if _is_term(a[k]) or _is_term(b[k]) or not isinstance(a[k], dict):
                nc += 1
            else:
                c2, a2, r2 = _count(a[k], b[k])
                nc += c2; na += a2; nr += r2
    return nc, na, nr


# ── main ──────────────────────────────────────────────────────────────────────

class _Loader(yaml.SafeLoader):
    pass


def _unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return tag_suffix


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _unknown)


def main() -> None:
    global USE_COLOR
    parser = argparse.ArgumentParser(description="Diff two config YAML files.")
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        USE_COLOR = False

    pa, pb = Path(args.file_a), Path(args.file_b)
    for p in (pa, pb):
        if not p.exists():
            print(f"[ERROR] not found: {p}", file=sys.stderr); sys.exit(1)

    da = yaml.load(pa.read_text(), Loader=_Loader) or {}
    db = yaml.load(pb.read_text(), Loader=_Loader) or {}

    nc, na, nr = _count(da, db)
    total = nc + na + nr

    W = 57
    print(_c(BOLD, f"\n{'─' * W}"))
    print(f"  {_c(DIM, 'A:')} {pa.name}")
    print(f"  {_c(DIM, 'B:')} {pb.name}")
    print(_c(BOLD, f"{'─' * W}"))
    if total == 0:
        print(_c(DIM, "  (identical)"))
    else:
        print(f"  {_c(YELLOW, f'~ {nc} changed')}   {_c(GREEN, f'+ {na} added')}   {_c(RED, f'- {nr} removed')}")
    print()

    # Print section by section so we get a blank line between top-level sections
    top_keys = list(dict.fromkeys(list(da) + list(db)))
    for key in top_keys:
        av, bv = da.get(key), db.get(key)
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


if __name__ == "__main__":
    main()
