"""Part 2 — Policy training (Jason).

`PolicyProtocol` is the typing contract used by `collect_rollouts` and the
auditor. `train()` and `load_policy()` are the real entry points implemented
in `ppo.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from reward_auditor.policy.ppo import PPOConfig, load_policy, train


@runtime_checkable
class PolicyProtocol(Protocol):
    """A policy is anything callable as `policy(obs) → action`.

    `obs` is the observation returned by `env.step`; `action` should match
    `env.action_space`. Random and PPO policies both satisfy this protocol.
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


# Type alias for convenience (functions are policies too).
PolicyLike = PolicyProtocol | Callable[[np.ndarray], np.ndarray]

__all__ = ["PPOConfig", "PolicyLike", "PolicyProtocol", "load_policy", "train"]
