"""Shim: ResidualMLP / ResidualPointNet now live in ``uwlab_rl.networks.point_net``
(single source of truth).

Kept so the BC pipeline's ``from residual_point_net import ResidualPointNet`` /
``from residual_point_net import ResidualMLP`` keep working while the online rsl_rl
DAgger student imports the exact same classes. Edit the architecture in
``source/uwlab_rl/uwlab_rl/networks/point_net.py``, not here.
"""

from uwlab_rl.networks.point_net import PointNet, ResidualMLP, ResidualPointNet  # noqa: F401
