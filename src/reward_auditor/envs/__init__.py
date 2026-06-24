"""Reward components, the reward wrapper, the task registry, and `make_env`."""

from __future__ import annotations

from reward_auditor.envs.components import (
    AliveBonus,
    ContactForcePenalty,
    EnergyCost,
    ForwardDisplacement,
    ForwardVelocity,
    HealthyPose,
    RewardComponent,
)
from reward_auditor.envs.make_env import make_env
from reward_auditor.envs.registry import (
    TaskSpec,
    available_components,
    get_task_spec,
    list_tasks,
)
from reward_auditor.envs.reward_wrapper import ComponentRewardWrapper

__all__ = [
    "AliveBonus",
    "ComponentRewardWrapper",
    "ContactForcePenalty",
    "EnergyCost",
    "ForwardDisplacement",
    "ForwardVelocity",
    "HealthyPose",
    "RewardComponent",
    "TaskSpec",
    "available_components",
    "get_task_spec",
    "list_tasks",
    "make_env",
]
