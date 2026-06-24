"""Tests for `ComponentRewardWrapper` — uses the stub env, no MuJoCo."""

from __future__ import annotations

import numpy as np
import pytest

from reward_auditor.envs.components import EnergyCost, ForwardVelocity
from reward_auditor.envs.reward_wrapper import ComponentRewardWrapper


def test_weighted_sum_matches_manual(stub_env_factory):
    env = ComponentRewardWrapper(
        stub_env_factory(),
        components=[ForwardVelocity(), EnergyCost(coef=1.0)],
        weights={"forward_velocity": 2.0, "energy_cost": -0.5},
    )
    obs, _ = env.reset(seed=0)
    action = np.array([0.5, 0.5], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    fv = info["reward_components"]["forward_velocity"]
    ec = info["reward_components"]["energy_cost"]
    expected = 2.0 * fv + (-0.5) * ec
    assert reward == pytest.approx(expected)


def test_info_populated_with_all_keys(stub_env_factory):
    env = ComponentRewardWrapper(
        stub_env_factory(),
        components=[ForwardVelocity()],
        weights={"forward_velocity": 1.0},
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([0.1, 0.1], dtype=np.float32))
    assert "reward_components" in info
    assert "reward_weights" in info
    assert "native_reward" in info
    assert "terminated" in info
    assert set(info["reward_components"]) == {"forward_velocity"}


def test_unknown_weight_key_raises(stub_env_factory):
    with pytest.raises(ValueError, match="no matching component"):
        ComponentRewardWrapper(
            stub_env_factory(),
            components=[ForwardVelocity()],
            weights={"nonexistent_component": 1.0},
        )


def test_duplicate_component_names_raise(stub_env_factory):
    with pytest.raises(ValueError, match="Duplicate"):
        ComponentRewardWrapper(
            stub_env_factory(),
            components=[ForwardVelocity(), ForwardVelocity()],
            weights={"forward_velocity": 1.0},
        )


def test_missing_weight_defaults_to_zero(stub_env_factory):
    env = ComponentRewardWrapper(
        stub_env_factory(),
        components=[ForwardVelocity(), EnergyCost(coef=1.0)],
        weights={"forward_velocity": 1.0},  # energy_cost omitted → weight 0
    )
    env.reset()
    _, reward, _, _, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    # energy_cost contribution should be 0 (weight 0), so reward == fv * 1
    assert reward == pytest.approx(info["reward_components"]["forward_velocity"])


def test_terminate_on_unhealthy_false_masks_termination(stub_env_factory):
    env = ComponentRewardWrapper(
        stub_env_factory(terminate_at=2),
        components=[ForwardVelocity()],
        weights={"forward_velocity": 1.0},
        terminate_on_unhealthy=False,
    )
    env.reset()
    for _ in range(5):
        _, _, term, _, _ = env.step(np.array([0.1, 0.1], dtype=np.float32))
        assert term is False, "V5 should never report terminated"


def test_terminate_on_unhealthy_true_passes_termination_through(stub_env_factory):
    env = ComponentRewardWrapper(
        stub_env_factory(terminate_at=2),
        components=[ForwardVelocity()],
        weights={"forward_velocity": 1.0},
        terminate_on_unhealthy=True,
    )
    env.reset()
    terms = [env.step(np.array([0.1, 0.1], dtype=np.float32))[2] for _ in range(3)]
    assert terms == [False, True, True]
