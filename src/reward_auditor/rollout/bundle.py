"""`RolloutBundle` — everything needed to build an `AuditInput` from one episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reward_auditor.auditor.schemas import AuditInput


@dataclass
class RolloutBundle:
    """One episode's worth of data on disk + in memory."""

    task: str
    variant: str
    weights: dict[str, float]
    video_path: Path
    states: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, act_dim)
    rewards: np.ndarray  # (T,)
    component_log: dict[str, np.ndarray]  # name → (T,)
    episode_return: float
    episode_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def to_audit_input(
    bundle: RolloutBundle,
    task_goal: str,
    available_components: list[str],
    include_reward_log: bool = True,
    include_component_log: bool = True,
) -> AuditInput:
    """Build an `AuditInput` from a `RolloutBundle`.

    The flags let callers (especially Part 3's modality ablation) drop signals.
    """
    return AuditInput(
        video_path=str(bundle.video_path),
        task_goal=task_goal,
        available_components=list(available_components),
        current_weights=dict(bundle.weights),
        reward_log=bundle.rewards.tolist() if include_reward_log else None,
        component_log=(
            {k: v.tolist() for k, v in bundle.component_log.items()}
            if include_component_log
            else None
        ),
    )
