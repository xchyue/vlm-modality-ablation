"""`collect_rollouts` — run a policy, capture states/actions/rewards/video, write to disk.

Policy is any callable obs → action (see `reward_auditor.policy.PolicyProtocol`).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from reward_auditor.rollout.bundle import RolloutBundle
from reward_auditor.rollout.render import render_mp4


def _is_rgb_array_env(env: gym.Env) -> bool:
    """Detect whether the underlying env was created with `render_mode='rgb_array'`."""
    rm = getattr(env, "render_mode", None)
    if rm is None and hasattr(env, "unwrapped"):
        rm = getattr(env.unwrapped, "render_mode", None)
    return rm == "rgb_array"


def collect_rollouts(
    policy: Callable[[np.ndarray], np.ndarray],
    env_fn: Callable[[], gym.Env],
    n_episodes: int,
    out_dir: Path,
    seed: int = 0,
    video_fps: int = 30,
    extra_metadata: dict[str, Any] | None = None,
) -> list[RolloutBundle]:
    """Run `n_episodes` of `policy` on `env_fn()` and dump everything to `out_dir`.

    Each episode produces `out_dir/episode_{i}/{video.mp4, trajectory.npz, meta.json}`.
    The env should be created with `render_mode='rgb_array'` for video; without it,
    no video is written but the rest of the bundle is still produced.

    Returns the in-memory bundles (same data as on disk, for downstream code that
    doesn't want to re-read files).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = env_fn()
    has_video = _is_rgb_array_env(env)
    bundles: list[RolloutBundle] = []

    try:
        for ep in range(n_episodes):
            ep_dir = out_dir / f"episode_{ep}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            obs, info = env.reset(seed=seed + ep)
            frames: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            component_log: dict[str, list[float]] = {}

            done = False
            steps = 0
            while not done:
                action = np.asarray(policy(obs), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)

                if has_video:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))

                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(action)
                rewards.append(float(reward))
                for name, val in info.get("reward_components", {}).items():
                    component_log.setdefault(name, []).append(float(val))

                steps += 1
                done = terminated or truncated

            states_arr = np.stack(states) if states else np.empty((0,), dtype=np.float32)
            actions_arr = np.stack(actions) if actions else np.empty((0,), dtype=np.float32)
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            comp_arr = {k: np.asarray(v, dtype=np.float32) for k, v in component_log.items()}

            video_path = ep_dir / "video.mp4"
            if has_video and frames:
                render_mp4(frames, video_path, fps=video_fps)

            # Pull task/variant/weights from the wrapper if present
            wrapper_weights = info.get("reward_weights", {})
            task_name = extra_metadata.get("task", "unknown") if extra_metadata else "unknown"
            variant_name = extra_metadata.get("variant", "unknown") if extra_metadata else "unknown"
            video_resolution: list[int] | None = None
            if frames:
                h, w = frames[0].shape[:2]
                video_resolution = [int(h), int(w)]
            meta: dict[str, Any] = {
                "task": task_name,
                "variant": variant_name,
                "weights": dict(wrapper_weights),
                "available_components": sorted(comp_arr),
                "episode_return": float(rewards_arr.sum()) if rewards_arr.size else 0.0,
                "episode_length": int(steps),
                "seed": int(seed + ep),
                "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "policy_ckpt": None,
                "gym_id": getattr(env.unwrapped.spec, "id", None) if env.spec else None,
                "video_path": "video.mp4" if has_video and frames else None,
                "video_fps": video_fps,
                "video_resolution": video_resolution,
            }
            if extra_metadata:
                meta.update(extra_metadata)

            # Save trajectory.npz
            np.savez(
                ep_dir / "trajectory.npz",
                states=states_arr,
                actions=actions_arr,
                rewards=rewards_arr,
                **{f"component_{k}": v for k, v in comp_arr.items()},
            )
            with (ep_dir / "meta.json").open("w") as f:
                json.dump(meta, f, indent=2)

            bundles.append(
                RolloutBundle(
                    task=task_name,
                    variant=variant_name,
                    weights=dict(wrapper_weights),
                    video_path=video_path,
                    states=states_arr,
                    actions=actions_arr,
                    rewards=rewards_arr,
                    component_log=comp_arr,
                    episode_return=float(rewards_arr.sum()) if rewards_arr.size else 0.0,
                    episode_length=int(steps),
                    metadata=meta,
                )
            )
    finally:
        env.close()

    return bundles


def random_policy(env: gym.Env) -> Callable[[np.ndarray], np.ndarray]:
    """Convenience: a policy that samples uniformly from `env.action_space`.

    Useful before Part 2's PPO is ready, and for smoke-testing.
    """
    action_space = env.action_space

    def _act(_obs: np.ndarray) -> np.ndarray:
        return action_space.sample()

    return _act
