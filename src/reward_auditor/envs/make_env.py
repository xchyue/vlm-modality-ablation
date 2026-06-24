"""`make_env` — the public entry point used by Parts 2/3/4.

Resolves the task + variant configs, instantiates the underlying Gymnasium env,
and wraps it with `ComponentRewardWrapper`. Caller-supplied `weights` override the
variant's defaults — this is the hook Part 4's VLM auditor uses.
"""

from __future__ import annotations

import gymnasium as gym

from reward_auditor.envs.registry import get_task_spec
from reward_auditor.envs.reward_wrapper import ComponentRewardWrapper
from reward_auditor.variants.loader import load_variant, resolve_weights


def make_env(
    task: str,
    variant: str,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct a Gymnasium env wrapped with the component reward.

    Args:
        task: e.g., "halfcheetah", "hopper", "ant", "humanoid".
        variant: e.g., "v1_ground_truth", "v3_shaping". Must match a YAML file under
            `configs/variants/`.
        weights: optional override for variant's default weights. Use this in the
            VLM reweight loop (Part 4) to inject auditor-proposed θ′.
        seed: seed passed to `env.reset` on first reset (the wrapper preserves it).
        render_mode: passed to `gymnasium.make`. Use `"rgb_array"` to enable video collection.

    Returns:
        A Gymnasium env with `observation_space`, `action_space`, and the component
        reward wrapper. Per-step `info` dict contains `reward_components`,
        `reward_weights`, and `native_reward`.
    """
    spec = get_task_spec(task)
    var = load_variant(variant)
    final_weights = resolve_weights(task, var, override=weights)

    base = gym.make(spec.gym_id, render_mode=render_mode)
    components = spec.make_components()
    env = ComponentRewardWrapper(
        base,
        components=components,
        weights=final_weights,
        terminate_on_unhealthy=var.terminate_on_unhealthy,
    )

    if seed is not None:
        env.reset(seed=seed)
    return env
