"""YAML loaders for task and variant configs, with cross-validation against the registry.

Failing fast here saves Part 2 a long PPO debugging session caused by a typo'd
weight key that would silently default to 0.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from reward_auditor.envs.registry import available_components, get_task_spec, list_tasks
from reward_auditor.variants.schema import TaskConfig, Variant

# configs/ lives at the repo root, two levels up from this file's package dir.
# src/reward_auditor/variants/loader.py → parents[3] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_ROOT = _REPO_ROOT / "configs"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}, got {type(data).__name__}.")
    return data


def load_task_config(task: str, configs_root: Path | None = None) -> TaskConfig:
    """Load `configs/tasks/{task}.yaml` and validate weight keys against the registry."""
    # Validate task name against the registry first — gives a clearer error than
    # "file not found" if a caller types `make_env("not_a_task", ...)`.
    spec = get_task_spec(task)
    root = configs_root or CONFIGS_ROOT
    cfg = TaskConfig(**_load_yaml(root / "tasks" / f"{task}.yaml"))
    if cfg.name != task:
        raise ValueError(f"Task YAML name mismatch: file={task!r} but yaml says {cfg.name!r}.")
    if spec.gym_id != cfg.gym_id:
        raise ValueError(
            f"Task {task!r}: registry gym_id={spec.gym_id!r} but YAML gym_id={cfg.gym_id!r}."
        )
    allowed = set(available_components(task))
    bad = set(cfg.default_weights) - allowed
    if bad:
        raise ValueError(
            f"Task {task!r}: default_weights has unknown components {sorted(bad)}. "
            f"Allowed: {sorted(allowed)}."
        )
    return cfg


def load_variant(name: str, configs_root: Path | None = None) -> Variant:
    """Load `configs/variants/{name}.yaml`. Does *not* validate against a task —
    that happens in `resolve_weights(task, variant)`."""
    root = configs_root or CONFIGS_ROOT
    v = Variant(**_load_yaml(root / "variants" / f"{name}.yaml"))
    if v.name != name:
        raise ValueError(f"Variant YAML name mismatch: file={name!r} but yaml says {v.name!r}.")
    return v


def resolve_weights(
    task: str,
    variant: Variant,
    override: dict[str, float] | None = None,
) -> dict[str, float]:
    """Merge task defaults ← variant weights ← caller override; validate all keys.

    Precedence: override > variant.weights > task.default_weights.
    If `variant.zero_others=True`, the base is all-zeros (one entry per task component)
    instead of task defaults — used by V4 ("proxy-only") variants.
    All keys must reference components registered for the task; unknown keys raise.
    """
    task_cfg = load_task_config(task)
    allowed = set(available_components(task))

    if variant.zero_others:
        merged: dict[str, float] = dict.fromkeys(available_components(task), 0.0)
    else:
        merged = dict(task_cfg.default_weights)
    merged.update(variant.weights)
    if override:
        merged.update(override)

    unknown = set(merged) - allowed
    if unknown:
        raise ValueError(
            f"Weights reference unknown components {sorted(unknown)} for task {task!r}. "
            f"Allowed: {sorted(allowed)}."
        )
    return merged


def list_variants(configs_root: Path | None = None) -> list[str]:
    """Names of all `configs/variants/*.yaml` files."""
    root = configs_root or CONFIGS_ROOT
    variants_dir = root / "variants"
    if not variants_dir.exists():
        return []
    return sorted(p.stem for p in variants_dir.glob("*.yaml"))


def list_known_tasks() -> list[str]:
    """Convenience re-export so callers don't need to import from `envs.registry` too."""
    return list_tasks()
