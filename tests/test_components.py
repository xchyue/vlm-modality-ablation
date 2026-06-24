"""Unit tests for reward components — no MuJoCo needed."""

from __future__ import annotations

import numpy as np
import pytest

from reward_auditor.envs.components import (
    AliveBonus,
    ContactForcePenalty,
    EnergyCost,
    ForwardDisplacement,
    ForwardVelocity,
    HealthyPose,
)


def _state():
    return np.zeros(4, dtype=np.float32)


def _action(v: float = 0.5):
    return np.full((2,), v, dtype=np.float32)


def test_forward_velocity_reads_info():
    c = ForwardVelocity()
    assert c(_state(), _action(), _state(), {"x_velocity": 1.23}) == pytest.approx(1.23)
    assert c(_state(), _action(), _state(), {}) == 0.0  # missing key → 0


def test_energy_cost_is_nonpositive():
    c = EnergyCost(coef=1.0)
    val = c(_state(), _action(0.5), _state(), {})
    assert val <= 0
    # Two action dims at 0.5 → ‖a‖² = 0.5
    assert val == pytest.approx(-0.5)


def test_energy_cost_rejects_negative_coef():
    with pytest.raises(ValueError):
        EnergyCost(coef=-1.0)


def test_alive_bonus_zero_when_terminated():
    c = AliveBonus()
    assert c(_state(), _action(), _state(), {"terminated": False}) == 1.0
    assert c(_state(), _action(), _state(), {"terminated": True}) == 0.0
    # default (no key) treated as not-terminated
    assert c(_state(), _action(), _state(), {}) == 1.0


def test_healthy_pose_indicator():
    c = HealthyPose(z_min=1.0, z_max=2.0)
    assert c(_state(), _action(), _state(), {"z_position": 1.5}) == 1.0
    assert c(_state(), _action(), _state(), {"z_position": 0.5}) == 0.0
    assert c(_state(), _action(), _state(), {"z_position": 2.5}) == 0.0


def test_healthy_pose_rejects_bad_range():
    with pytest.raises(ValueError):
        HealthyPose(z_min=2.0, z_max=1.0)


def test_contact_force_penalty_zero_when_no_forces():
    c = ContactForcePenalty(coef=1.0)
    assert c(_state(), _action(), _state(), {}) == 0.0


def test_contact_force_penalty_nonpositive():
    c = ContactForcePenalty(coef=1.0)
    forces = np.array([1.0, 2.0, 2.0])  # ‖f‖² = 9
    val = c(_state(), _action(), _state(), {"contact_forces": forces})
    assert val == pytest.approx(-9.0)


def test_forward_displacement_carries_state():
    c = ForwardDisplacement()
    # First call: no prev → 0
    assert c(_state(), _action(), _state(), {"x_position": 1.0}) == 0.0
    # Second call: delta = 1.5 - 1.0 = 0.5
    assert c(_state(), _action(), _state(), {"x_position": 1.5}) == pytest.approx(0.5)
    # Reset clears state
    c.reset()
    assert c(_state(), _action(), _state(), {"x_position": 3.0}) == 0.0


def test_forward_displacement_no_info():
    c = ForwardDisplacement()
    assert c(_state(), _action(), _state(), {}) == 0.0
