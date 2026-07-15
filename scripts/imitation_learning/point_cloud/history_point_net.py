"""Shim: HistoryPointNet now lives in ``uwlab_rl.networks.history_point_net`` (single source
of truth).

Kept so the BC pipeline's ``from history_point_net import HistoryPointNet`` keeps working while
the online rsl_rl DAgger student (``uwlab_rl.rsl_rl.student_teacher_history_pointcloud``) imports
the exact same class. Edit the architecture in
``source/uwlab_rl/uwlab_rl/networks/history_point_net.py``, not here.
"""

from uwlab_rl.networks.history_point_net import HistoryPointNet  # noqa: F401
