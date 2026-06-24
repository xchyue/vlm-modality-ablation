"""Pydantic models for VLM-auditor input/output (x_i, y_i in the paper).

These are the **stable interface** to Part 4 and Part 3. Do not break compatibility
without notifying Yisheng and 小川. New optional fields are fine; renaming or removing
required fields is not.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AuditInput(BaseModel):
    """x_i — everything the auditor sees about one rollout.

    Optional fields (`frame_descriptions`, `reward_log`, `component_log`) let
    Part 3's modality ablation (M1-M5) toggle which signals the auditor receives.
    """

    model_config = ConfigDict(extra="forbid")

    video_path: str  # v_i: path to rollout mp4
    task_goal: str  # g: natural-language task description
    available_components: list[str]  # C_all: component names registered for this task
    current_weights: dict[str, float]  # θ_i: weight dict the policy was trained under

    # Modality knobs — `None` means "this modality is hidden from the auditor".
    frame_descriptions: list[str] | None = None  # pre-computed frame captions
    reward_log: list[float] | None = None  # per-step total reward
    component_log: dict[str, list[float]] | None = None  # per-step c_k values, keyed by name


class AuditOutput(BaseModel):
    """y_i — the auditor's verdict on a rollout."""

    model_config = ConfigDict(extra="forbid")

    task_success: bool  # ŝ_i: did the policy actually do the task?
    reward_hacking_detected: bool  # ĥ_i: is the policy gaming the reward?
    reason: str  # natural-language explanation
    next_reward_weights: dict[str, float]  # θ′_i: proposed new weights
    severity: int | None = Field(default=None, ge=1, le=5)  # optional 1–5
