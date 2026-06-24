"""Task registry: name → gym_id + component factory + healthy-z config.

Single source of truth for which components are available per task. The factory
returns *fresh* component instances each call so stateful components (e.g.
`ForwardDisplacement`) don't leak state across envs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from reward_auditor.envs.components import (
    AliveBonus,
    ContactForcePenalty,
    EnergyCost,
    ForwardDisplacement,
    ForwardVelocity,
    HealthyPose,
    RewardComponent,
)


@dataclass(frozen=True)
class TaskSpec:
    """Static description of a MuJoCo locomotion task."""

    name: str  # e.g., "halfcheetah"
    gym_id: str  # e.g., "HalfCheetah-v5"
    component_factory: Callable[[], list[RewardComponent]]
    default_max_episode_steps: int = 1000

    def make_components(self) -> list[RewardComponent]:
        return self.component_factory()

    def component_names(self) -> list[str]:
        return [c.name for c in self.make_components()]


def _halfcheetah_components() -> list[RewardComponent]:
    # HalfCheetah does not terminate by default; alive_bonus is trivially 1
    # (it's included so V5 doesn't have to special-case the registry).
    return [
        ForwardVelocity(),
        EnergyCost(coef=0.1),
        ForwardDisplacement(),
        AliveBonus(),
    ]


def _hopper_components() -> list[RewardComponent]:
    # Hopper healthy z range (per Gymnasium MuJoCo v5): default healthy_z_range is (0.7, ∞).
    return [
        ForwardVelocity(),
        EnergyCost(coef=0.001),
        AliveBonus(),
        HealthyPose(z_min=0.7, z_max=float("inf"), obs_z_index=0),
        ForwardDisplacement(),
    ]


def _ant_components() -> list[RewardComponent]:
    # Ant healthy z range default (0.2, 1.0). Contact forces are the canonical Ant regularizer.
    return [
        ForwardVelocity(),
        EnergyCost(coef=0.5),
        AliveBonus(),
        HealthyPose(z_min=0.2, z_max=1.0, obs_z_index=0),
        ContactForcePenalty(coef=5e-4),
        ForwardDisplacement(),
    ]


def _humanoid_components() -> list[RewardComponent]:
    # Humanoid healthy z range default (1.0, 2.0).
    return [
        ForwardVelocity(),
        EnergyCost(coef=0.1),
        AliveBonus(),
        HealthyPose(z_min=1.0, z_max=2.0, obs_z_index=0),
        ContactForcePenalty(coef=5e-7),
        ForwardDisplacement(),
    ]


_REGISTRY: dict[str, TaskSpec] = {
    "halfcheetah": TaskSpec(
        name="halfcheetah",
        gym_id="HalfCheetah-v5",
        component_factory=_halfcheetah_components,
    ),
    "hopper": TaskSpec(
        name="hopper",
        gym_id="Hopper-v5",
        component_factory=_hopper_components,
    ),
    "ant": TaskSpec(
        name="ant",
        gym_id="Ant-v5",
        component_factory=_ant_components,
    ),
    "humanoid": TaskSpec(
        name="humanoid",
        gym_id="Humanoid-v5",
        component_factory=_humanoid_components,
    ),
}


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)


def get_task_spec(name: str) -> TaskSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Available: {list_tasks()}")
    return _REGISTRY[name]


def available_components(task: str) -> list[str]:
    return get_task_spec(task).component_names()
