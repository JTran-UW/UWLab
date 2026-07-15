#!/usr/bin/env python
"""Small wandb helper: browse your projects/runs and diff two runs' configs.

Examples
--------
# List all projects under your default entity (with run counts):
python scripts/wandb_interface.py list

# List runs in a project (most-recently-created first):
python scripts/wandb_interface.py list --project my-project [--limit 40]

# Diff the configs of two runs. A run ref is "entity/project/run_id",
# or "project/run_id" (uses your default entity), or just "run_id" together
# with --project / --entity:
python scripts/wandb_interface.py diff entity/proj/abc123 entity/proj/def456
python scripts/wandb_interface.py diff abc123 def456 --project my-project

Notes
-----
- Run *id* is the short hash in the URL (.../runs/<id>), not the display name.
- Needs the `wandb` package and a login (`wandb login` or ~/.netrc). On this
  machine that means the `patlab` conda env.
"""

from __future__ import annotations

import argparse
import sys

try:
    import wandb
except ImportError:
    sys.exit("wandb is not importable. Activate the `patlab` env first:\n"
             "  conda activate patlab")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def get_api() -> "wandb.Api":
    api = wandb.Api()
    if not hasattr(api, "Api") and api is None:  # pragma: no cover - defensive
        sys.exit("Could not create wandb.Api(). Are you logged in? Try `wandb login`.")
    return api


def resolve_entity(api: "wandb.Api", entity: str | None) -> str:
    ent = entity or api.default_entity
    if not ent:
        sys.exit("No entity given and no default entity found. Pass --entity.")
    return ent


def resolve_run_path(ref: str, entity: str | None, project: str | None) -> str:
    """Turn a loose run reference into a full 'entity/project/run_id' path."""
    parts = ref.split("/")
    if len(parts) == 3:
        return ref
    if len(parts) == 2:  # project/run_id
        proj, run_id = parts
        if not entity:
            sys.exit(f"'{ref}' has no entity and --entity was not given.")
        return f"{entity}/{proj}/{run_id}"
    if len(parts) == 1:  # bare run_id
        if not (entity and project):
            sys.exit(f"Run '{ref}' needs both --entity and --project (or use "
                     f"'entity/project/run_id').")
        return f"{entity}/{project}/{ref}"
    sys.exit(f"Could not parse run reference '{ref}'.")


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict into dotted keys for easy diffing."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def cmd_list(args) -> None:
    api = get_api()
    entity = resolve_entity(api, args.entity)

    if args.project:
        path = f"{entity}/{args.project}"
        runs = api.runs(path, order="-created_at")
        print(f"Runs in {path} (newest first):\n")
        header = f"{'RUN ID':<12}  {'STATE':<10}  {'CREATED':<20}  NAME"
        print(header)
        print("-" * len(header))
        for i, run in enumerate(runs):
            if args.limit and i >= args.limit:
                print(f"... (stopped at --limit {args.limit})")
                break
            created = str(getattr(run, "created_at", "") or "")[:19]
            print(f"{run.id:<12}  {str(run.state):<10}  {created:<20}  {run.name}")
        return

    # No project -> list projects under the entity.
    projects = list(api.projects(entity))
    if not projects:
        print(f"No projects found under entity '{entity}'.")
        return
    print(f"Projects under '{entity}':\n")
    for p in sorted(projects, key=lambda x: x.name):
        print(f"  {p.name}")
    print(f"\n{len(projects)} project(s). "
          f"Use `list --project <name>` to see its runs.")


def cmd_diff(args) -> None:
    api = get_api()
    entity = args.entity or api.default_entity

    path_a = resolve_run_path(args.run_a, entity, args.project)
    path_b = resolve_run_path(args.run_b, entity, args.project)

    run_a = api.run(path_a)
    run_b = api.run(path_b)

    # config keys starting with '_' are wandb internals (e.g. _wandb).
    cfg_a = flatten({k: v for k, v in run_a.config.items() if not k.startswith("_")})
    cfg_b = flatten({k: v for k, v in run_b.config.items() if not k.startswith("_")})

    print(f"A = {path_a}   ({run_a.name})")
    print(f"B = {path_b}   ({run_b.name})\n")

    keys = sorted(set(cfg_a) | set(cfg_b))
    only_a, only_b, changed = [], [], []
    for k in keys:
        in_a, in_b = k in cfg_a, k in cfg_b
        if in_a and not in_b:
            only_a.append(k)
        elif in_b and not in_a:
            only_b.append(k)
        elif cfg_a[k] != cfg_b[k]:
            changed.append(k)

    if changed:
        print("~ CHANGED (key: A -> B)")
        for k in changed:
            print(f"    {k}: {cfg_a[k]!r} -> {cfg_b[k]!r}")
        print()
    if only_a:
        print("- ONLY IN A")
        for k in only_a:
            print(f"    {k}: {cfg_a[k]!r}")
        print()
    if only_b:
        print("+ ONLY IN B")
        for k in only_b:
            print(f"    {k}: {cfg_b[k]!r}")
        print()

    if not (changed or only_a or only_b):
        print("Configs are identical.")
    else:
        print(f"Summary: {len(changed)} changed, {len(only_a)} only in A, "
              f"{len(only_b)} only in B.")


# ----------------------------------------------------------------------------
# entrypoint
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None,
                        help="wandb entity (team/user). Defaults to your default entity.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list projects, or runs within a project")
    p_list.add_argument("--project", default=None, help="project name; lists its runs")
    p_list.add_argument("--limit", type=int, default=None, help="max runs to show")
    p_list.set_defaults(func=cmd_list)

    p_diff = sub.add_parser("diff", help="diff configs of two runs")
    p_diff.add_argument("run_a", help="run ref: 'entity/project/run_id' or 'run_id'")
    p_diff.add_argument("run_b", help="run ref: 'entity/project/run_id' or 'run_id'")
    p_diff.add_argument("--project", default=None,
                        help="project for bare run_ids")
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
