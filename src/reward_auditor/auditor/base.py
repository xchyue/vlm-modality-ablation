"""Abstract auditor interface — any VLM auditor implements this Protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from reward_auditor.auditor.schemas import AuditInput, AuditOutput


@runtime_checkable
class AuditorProtocol(Protocol):
    """Contract for any auditor (VLM, LLM, or rule-based).

    Implementations live wherever — `auditor/vlm.py`, `loop/*`, etc. — as long as
    they expose a single `audit(x) → y` method.
    """

    def audit(self, x: AuditInput) -> AuditOutput: ...
