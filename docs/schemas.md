# Output File Schemas

These are the contracts between Part 1 (rollout collection) and downstream consumers (Part 4 VLM auditor, Part 5 analysis). **Do not change without notifying Yisheng and Zhiyuan.**

## `data/rollouts/{task}_{variant}/episode_{N}/`

```
episode_0/
├── video.mp4          # rgb_array frames @ 30fps
├── trajectory.npz     # numpy arrays (see below)
└── meta.json          # JSON metadata (see below)
```

### `trajectory.npz`

NPZ archive with the following arrays:

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `states` | `(T, obs_dim)` | `float32` | observations from `env.step` |
| `actions` | `(T, act_dim)` | `float32` | actions taken by the policy |
| `rewards` | `(T,)` | `float32` | total per-step reward (Σ wₖ·cₖ) |
| `component_<name>` | `(T,)` | `float32` | per-step value of component `<name>` (one array per registered component) |

Load with `np.load("trajectory.npz")`.

### `meta.json`

```json
{
  "task": "halfcheetah",
  "variant": "v3_shaping",
  "weights": { "forward_velocity": 10.0, "energy_cost": -0.1, "alive_bonus": 1.0 },
  "available_components": ["forward_velocity", "energy_cost", "alive_bonus"],
  "episode_return": 1234.56,
  "episode_length": 1000,
  "seed": 0,
  "timestamp": "2026-05-15T14:32:00Z",
  "policy_ckpt": "data/policies/halfcheetah_v3_seed0.pt",
  "gym_id": "HalfCheetah-v5",
  "video_path": "video.mp4",
  "video_fps": 30,
  "video_resolution": [480, 480]
}
```

All fields are required except `policy_ckpt` (null when using a random policy).

## `data/audits/{task}_{variant}_{episode}_{auditor}.json`

Written by Part 4 auditors. One file per audit call.

```json
{
  "audit_input": {
    "video_path": "data/rollouts/halfcheetah_v3/episode_0/video.mp4",
    "task_goal": "make the cheetah run forward as fast as possible without falling",
    "available_components": ["forward_velocity", "energy_cost", "alive_bonus"],
    "current_weights": { "forward_velocity": 10.0, "energy_cost": -0.1, "alive_bonus": 1.0 },
    "frame_descriptions": null,
    "reward_log": [0.1, 0.2, ...],
    "component_log": { "forward_velocity": [0.1, ...], "energy_cost": [-0.001, ...] }
  },
  "audit_output": {
    "task_success": true,
    "reward_hacking_detected": true,
    "reason": "Cheetah is flipping forward via shaping exploit rather than running.",
    "next_reward_weights": { "forward_velocity": 1.0, "energy_cost": -0.1, "alive_bonus": 1.0 },
    "severity": 3
  },
  "auditor_name": "ClaudeVLMAuditor",
  "timestamp": "2026-05-15T14:35:00Z"
}
```

`audit_input` mirrors the `AuditInput` pydantic model; `audit_output` mirrors `AuditOutput`. Use `model.model_dump_json()` to serialize.
