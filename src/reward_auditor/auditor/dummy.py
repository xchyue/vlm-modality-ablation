"""DummyAuditor — placeholder until Part 4 ships real VLM auditors.

Returns schema-valid output without inspecting the rollout. Used by:
  - Part 1 integration tests
  - Parts 2/3 before Part 4 is ready
"""

from __future__ import annotations

import random

from reward_auditor.auditor.schemas import AuditInput, AuditOutput


class DummyAuditor:
    """Random-but-valid auditor. Useful for plumbing tests, not for science.

    Args:
        deterministic: if True, always returns the same canned response. If False,
            randomizes `task_success` / `reward_hacking_detected` for fuzzing.
        seed: rng seed when not deterministic.
    """

    def __init__(self, deterministic: bool = True, seed: int = 0) -> None:
        self.deterministic = deterministic
        self._rng = random.Random(seed)

    def audit(self, x: AuditInput) -> AuditOutput:
        if self.deterministic:
            return AuditOutput(
                task_success=True,
                reward_hacking_detected=False,
                reason="(dummy auditor: no analysis performed)",
                next_reward_weights=dict(x.current_weights),
            )
        return AuditOutput(
            task_success=self._rng.random() > 0.3,
            reward_hacking_detected=self._rng.random() < 0.4,
            reason="(dummy auditor: random output)",
            next_reward_weights=dict(x.current_weights),
            severity=self._rng.randint(1, 5),
        )
