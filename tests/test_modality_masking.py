"""Tests for modality ablation — Part 3 (Xiaochuan).

Guards the integrity of the ablation:
  - per-modality signal masking (so "code > video" can't be confounded by a leak)
  - the metric computations (specificity / balanced_accuracy were added by me)
  - the M5 frame-description path (this is the test that catches the pipeline bug
    where M5 silently dropped its captions)

DROP AT: tests/test_modality_masking.py    Run: uv run pytest tests/test_modality_ablation.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reward_auditor.auditor.dummy import DummyAuditor
from reward_auditor.eval.metrics import (
    ConfusionMatrix,
    ablation_summary,
    is_hacking_variant,
)
from reward_auditor.eval.modality_ablation import (
    Modality,
    build_audit_input,
    run_ablation,
)
from reward_auditor.rollout.bundle import RolloutBundle


def _fake_bundle(task: str = "halfcheetah", variant: str = "v4_blatant") -> RolloutBundle:
    """Minimal RolloutBundle for unit tests (no real RL needed)."""
    return RolloutBundle(
        task=task,
        variant=variant,
        weights={"forward_velocity": 1.0},
        video_path=Path("/tmp/fake.mp4"),
        states=np.zeros((10, 17), dtype=np.float32),
        actions=np.zeros((10, 6), dtype=np.float32),
        rewards=np.arange(10, dtype=np.float32),
        component_log={"forward_velocity": np.arange(10, dtype=np.float32)},
        episode_return=45.0,
        episode_length=10,
    )


# 1. Masking integrity

def test_m1_hides_everything_except_task_goal():
    x = build_audit_input(_fake_bundle(), "run forward", ["forward_velocity"], Modality.M1_GOAL_ONLY)
    assert x.task_goal == "run forward"
    assert x.video_path == ""
    assert x.available_components == []
    assert x.current_weights == {}
    assert x.reward_log is None
    assert x.frame_descriptions is None


def test_m2_shows_code_but_not_video():
    x = build_audit_input(_fake_bundle(), "run forward", ["forward_velocity"], Modality.M2_PLUS_CODE)
    assert x.current_weights != {} # code/weights shown
    assert x.video_path == "" # video hidden


def test_m5_includes_frame_descriptions():
    # The regression test for the pipeline bug that dropped M5's captions.
    x = build_audit_input(
        _fake_bundle(), "run forward", ["forward_velocity"], Modality.M5_ALL,
        frame_descriptions=["frame0", "frame1"],
    )
    assert x.video_path == "/tmp/fake.mp4"
    assert x.reward_log is not None
    assert x.frame_descriptions == ["frame0", "frame1"]


# 2. Labels + metrics

def test_is_hacking_variant():
    assert is_hacking_variant("v1_ground_truth") is False
    assert is_hacking_variant("v4_blatant") is True


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix(tp=8, fp=2, tn=7, fn=3)
    assert cm.total == 20
    assert cm.accuracy == 0.75
    assert cm.precision == 0.8
    assert cm.recall == 8 / 11
    assert cm.specificity == 7 / 9
    assert round(cm.balanced_accuracy, 3) == 0.753


# 3. Pipeline

def test_run_ablation_produces_bundles_times_modalities():
    bundles = [_fake_bundle(variant="v1_ground_truth"), _fake_bundle(variant="v4_blatant")]
    results = run_ablation(
        bundles=bundles,
        auditor=DummyAuditor(),
        task_goals={"halfcheetah": "run forward"},
        available_components={"halfcheetah": ["forward_velocity"]},
    )
    assert len(results) == 2 * len(Modality)
    assert {r.modality for r in results} == set(Modality)


def test_ablation_summary_returns_all_modalities():
    results = run_ablation(
        bundles=[_fake_bundle(variant="v1_ground_truth")],
        auditor=DummyAuditor(),
        task_goals={"halfcheetah": "run forward"},
        available_components={"halfcheetah": ["forward_velocity"]},
    )
    summary = ablation_summary(results)
    assert set(summary.keys()) == {m.value for m in Modality}
    for entry in summary.values():
        assert "accuracy" in entry
        assert "per_variant" in entry

