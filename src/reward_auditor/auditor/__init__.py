"""VLM-auditor interface + dummy implementation.

This package defines the *contract* (`AuditorProtocol`, `AuditInput`, `AuditOutput`)
that Part 4 (Yisheng) implements with real VLM API calls. Part 1 ships:
  - the schemas (so all parts can agree on the data shape)
  - a `DummyAuditor` for testing and pre-Part-4 development
"""

from __future__ import annotations

from reward_auditor.auditor.base import AuditorProtocol
from reward_auditor.auditor.dummy import DummyAuditor
from reward_auditor.auditor.schemas import AuditInput, AuditOutput
from reward_auditor.auditor.vlm import VLMClientConfig, VLMAuditor

__all__ = [
    "AuditInput",
    "AuditOutput",
    "AuditorProtocol",
    "DummyAuditor",
    "VLMClientConfig",
    "VLMAuditor",
]
