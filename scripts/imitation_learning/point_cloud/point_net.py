"""Shim: PointNet now lives in ``uwlab_rl.networks.point_net`` (single source of truth).

Kept so the BC pipeline's ``from point_net import PointNet`` keeps working while the
online rsl_rl DAgger student imports the exact same class. Edit the architecture in
``source/uwlab_rl/uwlab_rl/networks/point_net.py``, not here.
"""

from uwlab_rl.networks.point_net import MLP, PointNet, ResidualMLP, ResidualPointNet  # noqa: F401
