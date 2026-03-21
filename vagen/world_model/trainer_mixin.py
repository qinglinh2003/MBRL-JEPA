"""Placeholder trainer integration for the GPT JEPA world model implementation.

The current implementation keeps the world model core independent from the PPO
trainer. This mixin is intentionally lightweight and can be extended once the
world model interfaces stabilize across GPT and Claude implementations.
"""

from typing import Dict


class WorldModelTrainerMixin:
    def _wm_init(self) -> None:
        self.wm_manager = None

    def _wm_post_rollout(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def _wm_update(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def _wm_get_metrics(self) -> Dict[str, float]:
        return {}
