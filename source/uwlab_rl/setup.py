# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'uwlab_rl' python package."""

import itertools
import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    #
    # Upper bound added 2026-08-28. rsl_rl's WandbSummaryWriter constructs
    # `wandb.Settings(start_method="thread")` (rsl_rl/utils/wandb_utils.py).
    # `start_method` was removed from wandb's Settings model, and because that
    # model forbids extra fields, newer wandb raises a pydantic ValidationError
    # ("Extra inputs are not permitted") the moment training starts logging --
    # after env creation and model construction, so it wastes a full startup.
    # 0.19.11 is the newest release that still accepts it.
    "wandb>=0.19.6,<0.20",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Extra dependencies for RL agents
#
# Pinned to a commit, not @main. An unpinned git dependency means a rebuild
# months later silently installs a different API than the one this code was
# written against -- which is exactly how the IsaacLab/rsl-rl mismatch of
# 2026-08-15 cost a night of downtime.
#
# 2026-08-28: bumped for the IsaacLab 3.0-beta/Newton migration (env_isaaclab3).
# 92f01d711a is UW-Lab/rsl_rl's feature/manipulation branch tip. It carries the
# rsl-rl-lib >=5.0.1 API (PPO takes `optimizer`, PPO.construct_algorithm /
# cfg["actor"] schema) that IsaacLab 3.0-beta's isaaclab_rl requires, *and*
# UWLab's own additions that the tasks depend on.
#
# Do NOT use the vendor/leggedrobotics branch here: it is a clean mirror of
# upstream leggedrobotics/rsl_rl@v5.2.0, so it has the right API but none of
# UWLab's additions -- no gsde noise (0 files vs 6 here) and no DAgger/behavior
# cloning (0 files vs 6 here). The omnireset agent config uses both
# (noise_std_type="gsde" in Base_PPORunnerCfg, and Base_DAggerRunnerCfg), so
# pinning the mirror silently breaks those runs while cartpole still passes.
#
# Must stay consistent with the IsaacLab commit pinned in uwlab.sh -- bump both
# together. NOTE: env_uwlab (IsaacLab 2.3.2, rsl-rl 3.1.2) is not being kept in
# sync with this pin -- see uwlab.sh.
EXTRAS_REQUIRE = {
    "rsl-rl": [
        "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git"
        "@92f01d711a08bcabd6391a6fc132dcfb82661795",
    ],
}

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

# Installation operation
setup(
    name="uwlab_rl",
    author="UW Lab Project Developers",
    maintainer="UW Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    extras_require=EXTRAS_REQUIRE,
    packages=["uwlab_rl"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
