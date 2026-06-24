"""Tests for variant/task YAML loading and resolution."""

from __future__ import annotations

import pytest

from reward_auditor.envs.registry import available_components, list_tasks
from reward_auditor.variants.loader import (
    list_variants,
    load_task_config,
    load_variant,
    resolve_weights,
)
from reward_auditor.variants.schema import Variant


def test_all_task_yamls_load():
    for task in list_tasks():
        cfg = load_task_config(task)
        assert cfg.name == task
        # default_weights only reference registered components
        unknown = set(cfg.default_weights) - set(available_components(task))
        assert not unknown, f"{task}: unknown default_weight keys {unknown}"


def test_all_variant_yamls_load():
    names = list_variants()
    assert set(names) >= {
        "v1_ground_truth",
        "v2_subtle",
        "v3_shaping",
        "v4_blatant",
        "v5_sim_bug",
    }
    for n in names:
        v = load_variant(n)
        assert v.name == n
        assert 1 <= v.hacking_severity <= 5


def test_resolve_weights_halfcheetah_v1():
    v = load_variant("v1_ground_truth")
    w = resolve_weights("halfcheetah", v)
    # All keys must be valid halfcheetah components
    assert set(w) <= set(available_components("halfcheetah"))
    # forward_velocity should be present (it's in halfcheetah defaults)
    assert "forward_velocity" in w


def test_resolve_weights_v4_zeros_others():
    v = load_variant("v4_blatant")
    w = resolve_weights("halfcheetah", v)
    # Only forward_velocity is nonzero; everything else explicitly zero
    nonzero = {k for k, val in w.items() if val != 0.0}
    assert nonzero == {"forward_velocity"}


def test_resolve_weights_v5_keeps_terminate_flag():
    v = load_variant("v5_sim_bug")
    assert v.terminate_on_unhealthy is False


def test_resolve_weights_override_takes_precedence():
    v = load_variant("v1_ground_truth")
    w = resolve_weights("halfcheetah", v, override={"forward_velocity": 99.0})
    assert w["forward_velocity"] == 99.0


def test_resolve_weights_unknown_override_raises():
    v = load_variant("v1_ground_truth")
    with pytest.raises(ValueError, match="unknown components"):
        resolve_weights("halfcheetah", v, override={"nonsense_component": 1.0})


def test_bad_task_name_raises():
    v = load_variant("v1_ground_truth")
    with pytest.raises(KeyError):
        resolve_weights("not_a_task", v)


def test_variant_schema_rejects_empty_weights():
    with pytest.raises(ValueError):
        Variant(name="x", hacking_severity=1, weights={})


def test_variant_schema_rejects_severity_out_of_range():
    with pytest.raises(ValueError):
        Variant(name="x", hacking_severity=9, weights={"forward_velocity": 1.0})
