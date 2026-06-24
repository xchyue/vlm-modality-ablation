"""M1-M5 modality ablation pipeline — Part 3 (Xiaochuan).

This module is the heart of the modality ablation study: given rollout bundles
and an auditor, run each (bundle × modality) audit and collect structured results.

The 5 modalities (M1 -> M6) progressively expose more signals to the VLM auditor,
letting us attribute which input modality drives detection accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from reward_auditor.auditor.base import AuditorProtocol
from reward_auditor.auditor.schemas import AuditInput, AuditOutput
from reward_auditor.rollout.bundle import RolloutBundle


class Modality(str, Enum):
    """The five ablation modalities, ordered by information content"""

    M1_GOAL_ONLY = "M1"
    M2_PLUS_CODE = "M2"
    M3_PLUS_STATS = "M3"
    M4_PLUS_VIDEO = "M4"
    M5_ALL = "M5"
    M6_VIDEO_ONLY = "M6"


@dataclass(frozen=True)
class ModalityConfig:
    """Which signals are visible to the auditor under a given modality"""

    include_video: bool
    include_components_and_weights: bool
    include_stats: bool
    include_frame_descriptions: bool


# Mapping from each Modality to its signal config
# Adding new modalities requires only adding an entry here
MODALITY_CONFIGS: dict[Modality, ModalityConfig] = {
    Modality.M1_GOAL_ONLY: ModalityConfig(
        include_video=False,
        include_components_and_weights=False,
        include_stats=False,
        include_frame_descriptions=False,
    ),
    Modality.M2_PLUS_CODE: ModalityConfig(
        include_video=False,
        include_components_and_weights=True,
        include_stats=False,
        include_frame_descriptions=False,
    ),
    Modality.M3_PLUS_STATS: ModalityConfig(
        include_video=False,
        include_components_and_weights=True,
        include_stats=True,
        include_frame_descriptions=False,
    ),
    Modality.M4_PLUS_VIDEO: ModalityConfig(
        include_video=True,
        include_components_and_weights=True,
        include_stats=False,
        include_frame_descriptions=False,
    ),
    Modality.M5_ALL: ModalityConfig(
        include_video=True,
        include_components_and_weights=True,
        include_stats=True,
        include_frame_descriptions=True,
    ),
    Modality.M6_VIDEO_ONLY: ModalityConfig(
        include_video=True,                    
        include_components_and_weights=False,
        include_stats=False,                 
        include_frame_descriptions=False,
    ),
}


def build_audit_input(
    bundle: RolloutBundle,
    task_goal: str,
    available_components: list[str],
    modality: Modality,
    frame_descriptions: list[str] | None = None,
) -> AuditInput:
    """Construct an `AuditInput` with fields masked according to `modality`

    Args:
        bundle: The rollout to audit
        task_goal: Natural language task description
        available_components: All reward components registered for this task
        modality: Which signals to expose (M1-M5)
        frame_descriptions: Pre computed frame captions, used only by M5

    Returns:
        An `AuditInput` ready to pass into `auditor.audit(...)`
    """

    cfg = MODALITY_CONFIGS[modality]

    return AuditInput(
        video_path=str(bundle.video_path) if cfg.include_video else "",
        task_goal=task_goal,
        available_components=(available_components if cfg.include_components_and_weights else []),
        current_weights=(dict(bundle.weights) if cfg.include_components_and_weights else {}),
        reward_log=bundle.rewards.tolist() if cfg.include_stats else None,
        component_log=(
            {k: v.tolist() for k, v in bundle.component_log.items()} if cfg.include_stats else None
        ),
        frame_descriptions=(
            frame_descriptions
            if (cfg.include_frame_descriptions and frame_descriptions is not None)
            else None
        ),
    )


@dataclass
class AblationResult:
    """One (bundle × modality) outcome."""

    bundle_task: str
    bundle_variant: str
    modality: Modality
    audit_output: AuditOutput


def run_ablation(
    bundles: Iterable[RolloutBundle],
    auditor: AuditorProtocol,
    task_goals: dict[str, str],
    available_components: dict[str, list[str]],
    modalities: Iterable[Modality] = tuple(Modality),
    frame_descriptions_per_bundle: dict[int, list[str]] | None = None,
    output_dir: Path | None = None,
) -> list[AblationResult]:
    """Run `auditor` on each (bundle × modality) pair.

    Args:
        bundles: Rollouts to audit. Iterable to allow streaming from disk.
        auditor: Anything satisfying `AuditorProtocol` (`audit(x) -> y`).
        task_goals: Mapping ``task_name -> natural-language goal``.
        available_components: Mapping ``task_name -> registered component names``.
        modalities: Subset of M1-M5 to evaluate. Defaults to all 5.
        frame_descriptions_per_bundle: Optional pre-computed frame captions,
            keyed by bundle index. Used only by M5.
        output_dir: If given, write per-audit JSON files to this directory。

    Returns:
        A flat list of `AblationResult`, length ``len(bundles) * len(modalities)``.
    """
    results: list[AblationResult] = []

    for i, bundle in enumerate(bundles):
        frame_desc = (
            frame_descriptions_per_bundle.get(i)
            if frame_descriptions_per_bundle is not None
            else None
        )

        for modality in modalities:
            x = build_audit_input(
                bundle=bundle,
                task_goal=task_goals[bundle.task],
                available_components=available_components[bundle.task],
                modality=modality,
                frame_descriptions=frame_desc,
            )

            y = auditor.audit(x)

            results.append(
                AblationResult(
                    bundle_task=bundle.task,
                    bundle_variant=bundle.variant,
                    modality=modality,
                    audit_output=y,
                )
            )

            if output_dir is not None:
                fname = f"{bundle.task}_{bundle.variant}_ep{i}_{auditor.name}_{modality.value}.json"
                audit_file = output_dir / fname
                audit_file.parent.mkdir(parents=True, exist_ok=True)
                audit_file.write_text(
                    json.dumps(
                        {
                            "audit_input": x.model_dump(mode="json"),
                            "audit_output": y.model_dump(mode="json"),
                            "auditor_name": auditor.name,
                            "modality": modality.value,
                        },
                        indent=2,
                    )
                )

    return results
