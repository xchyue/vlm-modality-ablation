"""Pydantic models for variant + task YAML configs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Variant(BaseModel):
    """A reward variant — weight overrides + an optional sim-bug-exploit flag.

    `terminate_on_unhealthy` is the V5 hook: setting it False lets the agent collect
    `alive_bonus` forever (sim-bug exploit) without the env terminating.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    hacking_severity: int = Field(ge=1, le=5)
    weights: dict[str, float]
    terminate_on_unhealthy: bool = True
    zero_others: bool = False  # if True, every component NOT in `weights` is forced to 0

    @field_validator("weights")
    @classmethod
    def _weights_nonempty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("Variant must define at least one weight.")
        return v


class TaskConfig(BaseModel):
    """A task YAML — gym_id, default weights, episode length."""

    model_config = ConfigDict(extra="forbid")

    name: str
    gym_id: str
    default_weights: dict[str, float]
    max_episode_steps: int = Field(default=1000, ge=1)
