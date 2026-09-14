# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
UWLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

UWLAB_TASKS_METADATA = toml.load(os.path.join(UWLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = UWLAB_TASKS_METADATA["package"]["version"]

##
# Enable Isaac Sim extensions that UWLab imports from.
##

# Isaac Sim 6.0 (Isaac Lab 3.0) no longer enables `isaacsim.core.utils` and friends
# by default, so `import isaacsim.core.utils.<x>` raises ModuleNotFoundError even
# though the extensions still ship. They live under `isaacsim/exts/` and only join
# sys.path once the Kit extension manager enables them.
#
# This must run before `import_packages` below: that call walks *every* sub-package,
# so a single module importing one of these (e.g. the omnireset terminations term
# importing `isaacsim.core.utils.bounds`) otherwise breaks registration of all tasks.
#
# Safe to call repeatedly; enabling an already-enabled extension is a no-op. Kept
# best-effort so importing uwlab_tasks without a running SimulationApp fails on the
# real missing-app error rather than on this.
_REQUIRED_ISAACSIM_EXTENSIONS = (
    "isaacsim.core.utils",  # .bounds/.carb/.prims/.stage/.viewports/.extensions
    "isaacsim.util.debug_draw",  # omnireset termination visualization
)


def _enable_isaacsim_extensions() -> None:
    try:
        import omni.kit.app
    except ImportError:
        return
    app = omni.kit.app.get_app()
    if app is None:
        return
    manager = app.get_extension_manager()
    if manager is None:
        return
    for ext in _REQUIRED_ISAACSIM_EXTENSIONS:
        try:
            manager.set_extension_enabled_immediate(ext, True)
        except Exception:  # noqa: BLE001 - never block task registration on this
            pass


_enable_isaacsim_extensions()

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
