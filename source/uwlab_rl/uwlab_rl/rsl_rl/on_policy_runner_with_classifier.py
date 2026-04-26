# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner that also trains a success classifier owned by a MultiResetManager.

Wraps ``self.alg.update`` so that right after PPO's policy update, the runner calls
``reset_manager.classifier_update(...)`` with the pairs accumulated during the rollout.
Returned metrics are merged into the loss dict so they appear in TensorBoard under ``Loss/``.
"""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner


class OnPolicyRunnerWithClassifier(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        self.classifier_num_epochs = int(train_cfg.get("classifier_num_epochs", 4))
        self.classifier_minibatch_size = int(train_cfg.get("classifier_minibatch_size", 256))

        self._reset_manager = None  # looked up lazily (MultiResetManager uses lazy_init)
        self._inner_update = self.alg.update

        def update_with_classifier():
            loss_dict = self._inner_update()
            if self._reset_manager is None:
                self._reset_manager = self._find_reset_manager()
            if self._reset_manager is not None:
                metrics = self._reset_manager.classifier_update(
                    n_epochs=self.classifier_num_epochs,
                    minibatch_size=self.classifier_minibatch_size,
                )
                loss_dict.update(metrics)
            return loss_dict

        self.alg.update = update_with_classifier

    def _find_reset_manager(self):
        """Walk env.event_manager, return the first MultiResetManager with use_classifier=True."""
        base_env = self.env.unwrapped
        event_mgr = getattr(base_env, "event_manager", None)
        if event_mgr is None:
            return None

        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import MultiResetManager

        for mode_cfgs in event_mgr._mode_class_term_cfgs.values():
            for term_cfg in mode_cfgs:
                func = term_cfg.func
                if isinstance(func, MultiResetManager) and getattr(func, "use_classifier", False):
                    return func
        return None
