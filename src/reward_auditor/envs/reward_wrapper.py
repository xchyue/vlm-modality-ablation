"""`ComponentRewardWrapper` — replaces the env's native reward with Σ_k w_k · c_k.

Critical invariants (relied upon by Part 4 VLM auditor):
  - `info["reward_components"]` is always populated after `step()` (dict[name → float])
  - `info["reward_weights"]` is always populated (dict[name → weight])
  - `info["native_reward"]` is preserved for reference / debugging
  - `info["terminated"]` is mirrored from the step return so `AliveBonus` can read it
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from reward_auditor.envs.components import RewardComponent


class ComponentRewardWrapper(gym.Wrapper):
    """Replace env reward with a component-weighted sum.

    Args:
        env: base Gymnasium env (must support `reset`/`step` with the standard API).
        components: ordered list of `RewardComponent` instances. Names must be unique.
        weights: dict mapping component name → scalar weight. Keys must be a subset
            of `{c.name for c in components}`; missing keys default to 0.
        terminate_on_unhealthy: if False, the wrapper masks `terminated` to always be False
            on step. Used by V5 sim-bug variants to allow "alive forever" exploits.

    Raises:
        ValueError: if component names collide, or if a weight key has no matching component.
    """

    def __init__(
        self,
        env: gym.Env,
        components: list[RewardComponent],
        weights: dict[str, float],
        terminate_on_unhealthy: bool = True,
    ) -> None:
        super().__init__(env)
        names = [c.name for c in components]
        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate component names: {set(dupes)}")
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(
                f"Weight keys {sorted(unknown)} have no matching component. "
                f"Available: {sorted(names)}"
            )
        self.components = components
        self.weights: dict[str, float] = dict(weights)
        self.terminate_on_unhealthy = terminate_on_unhealthy
        self._prev_obs: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        for c in self.components:
            if hasattr(c, "reset"):
                c.reset()  # type: ignore[attr-defined]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, native_reward, terminated, truncated, info = self.env.step(action)

        # Sim-bug variant: optionally mask termination so AliveBonus never goes to 0.
        if not self.terminate_on_unhealthy:
            terminated = False

        # AliveBonus needs to read the (possibly masked) terminated flag from info.
        info["terminated"] = terminated

        prev = self._prev_obs if self._prev_obs is not None else obs
        component_vals: dict[str, float] = {}
        for c in self.components:
            component_vals[c.name] = float(c(prev, action, obs, info))

        reward = sum(self.weights.get(name, 0.0) * v for name, v in component_vals.items())

        info["reward_components"] = component_vals
        info["reward_weights"] = dict(self.weights)
        info["native_reward"] = float(native_reward)

        self._prev_obs = obs
        return obs, float(reward), terminated, truncated, info
