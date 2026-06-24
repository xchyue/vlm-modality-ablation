"""Integration tests for the rollout package.

The unit-level test uses the stub env (fast, no MuJoCo). The end-to-end test
(`test_full_audit_cycle`) is marked `slow` because it spins up a real MuJoCo env.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reward_auditor.auditor import AuditInput, AuditOutput, DummyAuditor
from reward_auditor.envs.components import EnergyCost, ForwardVelocity
from reward_auditor.envs.reward_wrapper import ComponentRewardWrapper
from reward_auditor.rollout.bundle import to_audit_input
from reward_auditor.rollout.collect import collect_rollouts, random_policy


def _make_stub_env_factory(stub_env_factory, terminate_at=3):
    def env_fn():
        base = stub_env_factory(terminate_at=terminate_at)
        return ComponentRewardWrapper(
            base,
            components=[ForwardVelocity(), EnergyCost(coef=0.1)],
            weights={"forward_velocity": 1.0, "energy_cost": -0.1},
        )

    return env_fn


def test_collect_writes_artifacts(tmp_path: Path, stub_env_factory, random_action_policy):
    env_fn = _make_stub_env_factory(stub_env_factory)
    bundles = collect_rollouts(
        policy=random_action_policy,
        env_fn=env_fn,
        n_episodes=2,
        out_dir=tmp_path,
        seed=0,
        extra_metadata={"task": "stub", "variant": "v1_ground_truth"},
    )
    assert len(bundles) == 2
    for i, b in enumerate(bundles):
        ep_dir = tmp_path / f"episode_{i}"
        assert (ep_dir / "trajectory.npz").exists()
        assert (ep_dir / "meta.json").exists()
        traj = np.load(ep_dir / "trajectory.npz")
        assert "states" in traj.files
        assert "actions" in traj.files
        assert "rewards" in traj.files
        # Per-component arrays prefixed `component_`
        assert any(f.startswith("component_") for f in traj.files)
        # Metadata is valid JSON with required keys
        meta = json.loads((ep_dir / "meta.json").read_text())
        for key in ["task", "variant", "weights", "episode_return", "episode_length", "seed"]:
            assert key in meta


def test_to_audit_input_with_full_modality(stub_env_factory, random_action_policy, tmp_path):
    env_fn = _make_stub_env_factory(stub_env_factory)
    bundles = collect_rollouts(
        policy=random_action_policy,
        env_fn=env_fn,
        n_episodes=1,
        out_dir=tmp_path,
        seed=0,
        extra_metadata={"task": "stub", "variant": "v1_ground_truth"},
    )
    x = to_audit_input(
        bundles[0],
        task_goal="run forward",
        available_components=["forward_velocity", "energy_cost"],
    )
    assert isinstance(x, AuditInput)
    assert x.reward_log is not None
    assert x.component_log is not None


def test_to_audit_input_with_subset_modality(stub_env_factory, random_action_policy, tmp_path):
    env_fn = _make_stub_env_factory(stub_env_factory)
    bundles = collect_rollouts(
        policy=random_action_policy,
        env_fn=env_fn,
        n_episodes=1,
        out_dir=tmp_path,
        seed=0,
        extra_metadata={"task": "stub", "variant": "v1_ground_truth"},
    )
    # M2: code + weights only, no stats
    x = to_audit_input(
        bundles[0],
        task_goal="run forward",
        available_components=["forward_velocity", "energy_cost"],
        include_reward_log=False,
        include_component_log=False,
    )
    assert x.reward_log is None
    assert x.component_log is None


def test_full_cycle_stub(stub_env_factory, random_action_policy, tmp_path):
    """Stub-env version of the README §4.7 integration cycle:

        make_env → collect_rollouts → to_audit_input → DummyAuditor.audit
        → env with auditor's next_reward_weights → collect again

    Uses the stub env so this runs fast in CI. The real MuJoCo version is
    in `test_full_audit_cycle_mujoco` (marked `slow`).
    """
    env_fn = _make_stub_env_factory(stub_env_factory)
    bundles = collect_rollouts(
        policy=random_action_policy,
        env_fn=env_fn,
        n_episodes=1,
        out_dir=tmp_path,
        seed=0,
        extra_metadata={"task": "stub", "variant": "v1_ground_truth"},
    )
    audit_input = to_audit_input(
        bundles[0],
        task_goal="run forward",
        available_components=["forward_velocity", "energy_cost"],
    )

    auditor = DummyAuditor()
    audit_output: AuditOutput = auditor.audit(audit_input)
    assert audit_output.next_reward_weights == audit_input.current_weights

    # Build a new env_fn using the auditor's weights (the Part 4 hook)
    def env_fn_v2():
        base = stub_env_factory(terminate_at=3)
        return ComponentRewardWrapper(
            base,
            components=[ForwardVelocity(), EnergyCost(coef=0.1)],
            weights=audit_output.next_reward_weights,
        )

    bundles_v2 = collect_rollouts(
        policy=random_action_policy,
        env_fn=env_fn_v2,
        n_episodes=1,
        out_dir=tmp_path / "v2",
        seed=0,
    )
    assert len(bundles_v2) == 1


@pytest.mark.slow
def test_full_audit_cycle_mujoco(tmp_path: Path):
    """End-to-end cycle on a real MuJoCo env. Skipped by default; run with `-m slow`."""
    pytest.importorskip("mujoco")
    from reward_auditor.envs import available_components, make_env

    def env_fn():
        return make_env("halfcheetah", "v1_ground_truth", seed=0, render_mode=None)

    env = env_fn()
    policy = random_policy(env)
    env.close()

    bundles = collect_rollouts(
        policy=policy,
        env_fn=env_fn,
        n_episodes=1,
        out_dir=tmp_path,
        seed=0,
        extra_metadata={"task": "halfcheetah", "variant": "v1_ground_truth"},
    )
    assert len(bundles) == 1
    assert bundles[0].episode_length > 0

    x = to_audit_input(
        bundles[0],
        task_goal="make the cheetah run forward",
        available_components=available_components("halfcheetah"),
    )
    auditor = DummyAuditor()
    y = auditor.audit(x)
    assert isinstance(y, AuditOutput)

    # Apply auditor's weights back into make_env — must not raise
    new_env = make_env("halfcheetah", "v1_ground_truth", weights=y.next_reward_weights, seed=0)
    new_env.close()
