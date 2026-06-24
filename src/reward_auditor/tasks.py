"""Task goals and filename helpers for the modality ablation benchmark.

Extracted from the main reward_auditor repo so this package does not depend on
MuJoCo, torch, or the Part 4 reweight loop.
"""

from __future__ import annotations

TASK_GOALS: dict[str, str] = {
    "halfcheetah": "make the half-cheetah run forward quickly with stable locomotion",
    "hopper": "make the one-legged hopper move forward while staying upright and controlled",
    "ant": "make the quadruped ant move forward with a stable gait",
    "humanoid": "make the humanoid run forward while remaining upright and balanced",
}


def default_task_goal(task: str) -> str:
    """Return the natural-language goal used in auditor prompts."""
    return TASK_GOALS.get(task, f"make the {task} agent complete its locomotion task")


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)


def available_components_from_meta(metadata: dict) -> list[str]:
    """Read registered reward component names from rollout meta.json."""
    components = metadata.get("available_components")
    if components:
        return list(components)
    raise KeyError("meta.json missing 'available_components'")
