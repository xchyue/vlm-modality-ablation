"""Named reward components — pure functions c_k(state, action, next_state, info) → float.

A `RewardComponent` is the smallest reusable unit of reward. `ComponentRewardWrapper`
combines them as Σ_k w_k · c_k. Components are intentionally simple: they read from the
MuJoCo env's `info` dict where possible and fall back to obs slicing otherwise.

See `docs/spec.md` for the canonical list. Adding a new component:
  1. Subclass `RewardComponent`, set `name` and implement `__call__`.
  2. Wire it into `envs/registry.py` for the relevant task(s).
  3. Add a row to the variant YAMLs that need it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

ObsLike = np.ndarray
ActionLike = np.ndarray
InfoDict = dict[str, Any]


class RewardComponent(ABC):
    """One named term in the reward sum.

    Subclasses must set `name` (globally unique within a task) and implement `__call__`.
    """

    name: str

    @abstractmethod
    def __call__(
        self,
        state: ObsLike,
        action: ActionLike,
        next_state: ObsLike,
        info: InfoDict,
    ) -> float:
        """Return this component's scalar value for the transition."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ForwardVelocity(RewardComponent):
    """Forward x-axis velocity, read from MuJoCo's `info["x_velocity"]`.

    Present in HalfCheetah/Hopper/Ant/Humanoid v4+ Gymnasium MuJoCo envs.
    """

    name = "forward_velocity"

    def __call__(self, state, action, next_state, info) -> float:
        return float(info.get("x_velocity", 0.0))


class EnergyCost(RewardComponent):
    """Quadratic control cost: −coef · ‖action‖²."""

    name = "energy_cost"

    def __init__(self, coef: float = 1.0) -> None:
        if coef < 0:
            raise ValueError("EnergyCost coef must be ≥ 0 (sign is applied at use site).")
        self.coef = coef

    def __call__(self, state, action, next_state, info) -> float:
        return -self.coef * float(np.square(np.asarray(action, dtype=np.float32)).sum())


class AliveBonus(RewardComponent):
    """+1 each step the env hasn't terminated. V5 exploits this when termination is disabled."""

    name = "alive_bonus"

    def __call__(self, state, action, next_state, info) -> float:
        # `info["terminated"]` is populated by `ComponentRewardWrapper` (see reward_wrapper.py)
        # so we can read it here. For tasks that never terminate (HalfCheetah), this is always 1.
        terminated = bool(info.get("terminated", False))
        return 0.0 if terminated else 1.0


class HealthyPose(RewardComponent):
    """Indicator that the env's torso height z ∈ [z_min, z_max].

    Reads `info["z_position"]` if present, otherwise falls back to `state[obs_z_index]`.
    For Gymnasium v5 MuJoCo envs (Hopper/Humanoid), the healthy z is exposed via
    `info["z_distance_from_origin"]` or directly via the observation; we accept an
    explicit `obs_z_index` so we don't have to hard-code env internals.
    """

    name = "healthy_pose"

    def __init__(
        self,
        z_min: float,
        z_max: float = float("inf"),
        obs_z_index: int | None = None,
    ) -> None:
        if z_min > z_max:
            raise ValueError(f"z_min ({z_min}) must be ≤ z_max ({z_max}).")
        self.z_min = z_min
        self.z_max = z_max
        self.obs_z_index = obs_z_index

    def _read_z(self, state: ObsLike, info: InfoDict) -> float:
        if "z_position" in info:
            return float(info["z_position"])
        if self.obs_z_index is not None:
            return float(np.asarray(state)[self.obs_z_index])
        # Default: most MuJoCo locomotion envs put torso z at obs[0] when `exclude_current_positions_from_observation=False`,
        # but the safer default is to assume the env has been configured to expose z via info.
        raise KeyError(
            "HealthyPose could not read z: pass `obs_z_index` or use an env that exposes `info['z_position']`."
        )

    def __call__(self, state, action, next_state, info) -> float:
        try:
            z = self._read_z(next_state, info)
        except KeyError:
            return 0.0
        return 1.0 if self.z_min <= z <= self.z_max else 0.0


class ContactForcePenalty(RewardComponent):
    """−coef · ‖contact_forces‖². Ant-specific (uses `info["contact_forces"]` or similar)."""

    name = "contact_force_penalty"

    def __init__(self, coef: float = 1.0) -> None:
        if coef < 0:
            raise ValueError("ContactForcePenalty coef must be ≥ 0.")
        self.coef = coef

    def __call__(self, state, action, next_state, info) -> float:
        forces = info.get("contact_forces")
        if forces is None:
            # Gymnasium v5 Ant exposes the sum-of-squares directly:
            cfrc = info.get("reward_contact")
            if cfrc is not None:
                # `reward_contact` is already negated; convert back to magnitude
                return -self.coef * float(abs(cfrc))
            return 0.0
        forces_arr = np.asarray(forces, dtype=np.float32)
        return -self.coef * float(np.square(forces_arr).sum())


class ForwardDisplacement(RewardComponent):
    """Per-step forward displacement: x_position(t) − x_position(t−1).

    Uses `info["x_position"]` if available; falls back to 0 on the first step (no prev_x).
    State is carried via the component itself (resets via `reset()`).
    """

    name = "forward_displacement"

    def __init__(self) -> None:
        self._prev_x: float | None = None

    def reset(self) -> None:
        """Called by `ComponentRewardWrapper.reset()` to clear stateful components."""
        self._prev_x = None

    def __call__(self, state, action, next_state, info) -> float:
        x = info.get("x_position")
        if x is None:
            return 0.0
        x = float(x)
        if self._prev_x is None:
            self._prev_x = x
            return 0.0
        delta = x - self._prev_x
        self._prev_x = x
        return delta
