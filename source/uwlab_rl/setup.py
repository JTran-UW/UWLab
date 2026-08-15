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
    "wandb>=0.19.6",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Extra dependencies for RL agents
#
# Pinned to a commit, not @main. An unpinned git dependency means a rebuild
# months later silently installs a different API than the one this code was
# written against -- which is exactly how the IsaacLab/rsl-rl mismatch of
# 2026-08-15 cost a night of downtime.
#
# 959ccbc4 is the rsl-rl 3.1.2 API (PPO takes no `optimizer` kwarg, no
# construct_algorithm/cfg["actor"] schema). It must stay consistent with the
# IsaacLab commit pinned in uwlab.sh -- bump both together.
EXTRAS_REQUIRE = {
    "rsl-rl": [
        "rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git"
        "@959ccbc4712400a75efaa63cb827fd8939776825",
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
