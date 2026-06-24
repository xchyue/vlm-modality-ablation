"""Rollout collection: run a policy in an env, capture (states, actions, rewards, video)."""

from __future__ import annotations

from reward_auditor.rollout.bundle import RolloutBundle, to_audit_input
from reward_auditor.rollout.collect import collect_rollouts, random_policy
from reward_auditor.rollout.render import render_mp4

__all__ = ["RolloutBundle", "collect_rollouts", "random_policy", "render_mp4", "to_audit_input"]
