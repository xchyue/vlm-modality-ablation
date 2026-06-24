"""Shared pytest fixtures.

Fast fixtures (no MuJoCo) are unconditional. MuJoCo-dependent fixtures live
behind the `slow` marker and are skipped by default — run with `pytest -m slow`
to include them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest


class _StubEnv(gym.Env):
    """Minimal Gymnasium env for unit tests, no MuJoCo needed."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, terminate_at: int | None = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,))
        self.spec = None
        self.render_mode = None
        self._terminate_at = terminate_at
        self._t = 0
        self._x = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._t = 0
        self._x = 0.0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        self._t += 1
        # fake velocity drawn from action[0] to make tests deterministic
        v = float(np.asarray(action).flatten()[0])
        self._x += v
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        terminated = self._terminate_at is not None and self._t >= self._terminate_at
        truncated = False
        info = {"x_velocity": v, "x_position": self._x, "z_position": 1.5}
        return obs, 0.0, terminated, truncated, info

    def render(self):  # pragma: no cover — not used in unit tests
        return None

    def close(self):  # pragma: no cover
        pass


@pytest.fixture
def stub_env_factory() -> Callable[..., _StubEnv]:
    """Build a `_StubEnv` with customizable shape/termination. Lightweight."""
    return _StubEnv


@pytest.fixture
def random_action_policy() -> Callable[[np.ndarray], np.ndarray]:
    """A deterministic 'random' policy (seeded) for reproducible tests."""
    rng = np.random.default_rng(42)

    def policy(_obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    return policy
