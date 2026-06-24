"""Smoketest for Part 1: env loads, takes N random steps, reward populated, mp4 written.

Usage:
    uv run python scripts/make_env_smoketest.py --task halfcheetah --variant v1_ground_truth
    uv run python scripts/make_env_smoketest.py --task halfcheetah --variant v4_blatant --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reward_auditor.envs import available_components, make_env
from reward_auditor.rollout.render import render_mp4


def main() -> int:
    p = argparse.ArgumentParser(description="Part 1 smoketest: env + reward wrapper.")
    p.add_argument("--task", required=True, help="halfcheetah | hopper | ant | humanoid")
    p.add_argument("--variant", required=True, help="v1_ground_truth | … | v5_sim_bug")
    p.add_argument("--steps", type=int, default=100, help="Number of random-action steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/smoketests"),
        help="Where to write the smoketest mp4.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rgb_array render (useful in headless CI without MuJoCo GL).",
    )
    args = p.parse_args()

    render_mode = None if args.no_video else "rgb_array"
    print(f"[smoketest] task={args.task} variant={args.variant} render_mode={render_mode}")
    print(f"[smoketest] available components: {available_components(args.task)}")

    env = make_env(args.task, args.variant, seed=args.seed, render_mode=render_mode)
    print(f"[smoketest] obs_space={env.observation_space}  act_space={env.action_space}")

    obs, info = env.reset(seed=args.seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    component_sums: dict[str, float] = {}

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        for k, v in info["reward_components"].items():
            component_sums[k] = component_sums.get(k, 0.0) + float(v)
        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        if terminated or truncated:
            print(
                f"[smoketest] episode ended at step {t + 1} "
                f"(terminated={terminated} truncated={truncated})"
            )
            break

    # Invariant checks
    assert "reward_components" in info, "wrapper missed reward_components"
    assert "reward_weights" in info, "wrapper missed reward_weights"
    assert "native_reward" in info, "wrapper missed native_reward"

    print(f"[smoketest] total reward over {args.steps} steps: {total_reward:.4f}")
    print("[smoketest] per-component sums:")
    for k in sorted(component_sums):
        weight = info["reward_weights"].get(k, 0.0)
        print(f"           {k:30s}  Σc={component_sums[k]:+10.4f}  w={weight:+.3f}")

    if frames:
        out_path = args.video_dir / f"{args.task}_{args.variant}.mp4"
        render_mp4(frames, out_path, fps=30)
        print(f"[smoketest] wrote {len(frames)}-frame mp4 → {out_path}")
    elif not args.no_video:
        print("[smoketest] WARNING: render_mode='rgb_array' but no frames captured.")

    env.close()
    print("[smoketest] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
