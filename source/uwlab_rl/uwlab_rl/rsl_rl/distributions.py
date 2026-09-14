# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration distribution reproducing pre-3.0 OmniReset training behavior.

The OmniReset configs were written against a rsl-rl whose ``gsde`` option was
not gSDE. Upstream commit 3030b45 ("Add demo proving main's gSDE is
heteroscedastic") showed that implementation built the correct marginal Normal
but never applied ``phi @ epsilon``, and resampled the weight matrix only once
in ``__init__`` -- so it sampled straight from the marginal, giving noise that
was independent across both time and environments.

rsl-rl 5.x implements gSDE correctly, holding the weight matrix fixed for a
whole rollout (``OnPolicyRunner``). That is the better algorithm in general, but
it is a different one, and contact-rich insertion was tuned against the old
behavior: measured on a trained OmniReset actor in simulation, lag-1 noise
autocorrelation goes from -0.001 to +0.86 and step-to-step noise variation drops
3.4x, which suppresses the contact discovery the task depends on.
"""

from __future__ import annotations

import torch

from rsl_rl.modules.distribution import GsdeDistribution


class LegacyGsdeDistribution(GsdeDistribution):
    """gSDE parameterization, sampled the way the pre-3.0 code sampled it.

    Keeps everything about :class:`~rsl_rl.modules.distribution.GsdeDistribution`
    -- the 2-D ``log_std`` of shape ``(latent_dim, output_dim)``, the
    state-dependent stddev ``sqrt(phi(s)**2 @ exp(log_std)**2)``, log-probs,
    entropy, and checkpoint layout -- and changes only how actions are drawn.

    Instead of ``mean + phi(s) @ W`` with ``W`` frozen for a rollout, this draws
    from the marginal ``Normal(mean, stddev)`` directly. That makes the noise
    independent across both timesteps and environments while leaving the
    distribution it is drawn from bit-for-bit identical, which is exactly what
    the pre-3.0 implementation did.

    Drawing per-environment matters as much as drawing per-step: a single ``W``
    shared across the batch correlates the noise between environments even when
    resampled every step, which measurably distorts the per-action noise
    statistics (+-5% vs +-0.2% on normalized per-action std).

    Nothing here affects the PPO objective -- the weight matrix never entered
    log-probs or entropy, so only the sampled actions change.
    """

    def sample(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("LegacyGsdeDistribution.sample called before update().")
        return self._distribution.sample()
