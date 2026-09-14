#!/usr/bin/env python3
"""Apply the rsl_rl gSDE activation fix to an rsl_rl package directory (idempotent).

Bug (UW-Lab/rsl_rl feature/manipulation @ 92f01d71): ``MLP.__init__`` appends one shared
activation module after every hidden layer, and the gSDE branch of ``MLPModel.forward``
walks ``list(self.mlp.children())``. ``children()`` de-duplicates modules by identity, so
the walk keeps only the first activation and the trained policy is
``Linear -> act -> Linear -> Linear -> ... -> Linear`` instead of the configured MLP.
See ISAACLAB_3_GRASP_HANDOFF.md section 10.

Usage:
    apply_rsl_rl_gsde_fix.py <path/to/rsl_rl>        # patch in place, exit 0 if fixed
    apply_rsl_rl_gsde_fix.py --check <path/to/rsl_rl> # exit 0 if already fixed, 1 otherwise
"""

import sys
from pathlib import Path

OLD = "            children = list(self.mlp.children())\n"
NEW = (
    "            # ``MLP`` appends one shared activation instance after every hidden\n"
    "            # layer; ``children()`` de-duplicates modules by identity and would\n"
    "            # drop all but the first activation. Iterate the Sequential itself.\n"
    "            children = list(self.mlp)\n"
)
FIXED_MARKER = "            children = list(self.mlp)\n"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    target = Path(args[0]) / "models" / "mlp_model.py"
    if not target.exists():
        print(f"[gsde-fix] not an rsl_rl package dir: {args[0]}")
        return 2
    src = target.read_text()
    if FIXED_MARKER in src:
        print(f"[gsde-fix] already applied: {target}")
        return 0
    if check_only:
        print(f"[gsde-fix] NOT applied: {target}")
        return 1
    if src.count(OLD) != 1:
        print(f"[gsde-fix] unexpected source (found {src.count(OLD)} match(es) for the buggy line): {target}")
        return 1
    target.write_text(src.replace(OLD, NEW))
    print(f"[gsde-fix] applied: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
