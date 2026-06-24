"""Collect N episodes of rollouts and write videos + trajectories + metadata.

Usage:
    # With a trained policy (Part 2 ships .pt checkpoints):
    uv run python scripts/collect_rollouts.py \\
        --task halfcheetah --variant v3_shaping \\
        --policy data/policies/halfcheetah_v3_seed0.pt \\
        --n-episodes 5 --out-dir data/rollouts/halfcheetah_v3/

    # Without a policy (random-action smoke run):
    uv run python scripts/collect_rollouts.py \\
        --task halfcheetah --variant v1_ground_truth \\
        --n-episodes 1 --out-dir data/rollouts/halfcheetah_v1_random/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reward_auditor.envs import make_env
from reward_auditor.rollout.collect import collect_rollouts, random_policy


def _load_policy(policy_path: Path | None, env):
    if policy_path is None:
        return random_policy(env)
    from reward_auditor.policy import load_policy

    return load_policy(policy_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Collect rollouts and write to disk.")
    p.add_argument("--task", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--policy", type=Path, default=None, help="Path to .pt checkpoint (Part 2).")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    render_mode = None if args.no_video else "rgb_array"
    env = make_env(args.task, args.variant, seed=args.seed, render_mode=render_mode)
    policy = _load_policy(args.policy, env)
    env.close()  # collect_rollouts will rebuild via env_fn

    def env_fn():
        return make_env(args.task, args.variant, seed=args.seed, render_mode=render_mode)

    bundles = collect_rollouts(
        policy=policy,
        env_fn=env_fn,
        n_episodes=args.n_episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        extra_metadata={
            "task": args.task,
            "variant": args.variant,
            "policy_ckpt": str(args.policy) if args.policy else None,
        },
    )

    print(f"[collect] wrote {len(bundles)} episodes → {args.out_dir}")
    for i, b in enumerate(bundles):
        print(
            f"  episode_{i}: T={b.episode_length}  return={b.episode_return:.3f}  "
            f"video={'yes' if b.video_path.exists() else 'no'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
