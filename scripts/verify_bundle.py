"""Verify benchmark bundle completeness and optionally compare golden metrics.

Usage:
    uv run python scripts/verify_bundle.py
    uv run python scripts/verify_bundle.py --check-metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_TASKS = ("ant", "halfcheetah", "hopper", "humanoid")
EXPECTED_VARIANTS = (
    "v1_ground_truth",
    "v2_subtle",
    "v3_shaping",
    "v4_blatant",
    "v5_sim_bug",
)
EXPECTED_EPISODES = 3
EXPECTED_AUDITS = 360
METRIC_TOLERANCE = 0.05


def _count_rollouts(rollout_dir: Path) -> tuple[int, list[str]]:
    missing: list[str] = []
    found = 0
    for task in EXPECTED_TASKS:
        for variant in EXPECTED_VARIANTS:
            base = rollout_dir / f"{task}_{variant}_seed0" / "iter_0"
            for ep in range(EXPECTED_EPISODES):
                ep_dir = base / f"episode_{ep}"
                if (ep_dir / "trajectory.npz").exists():
                    found += 1
                else:
                    missing.append(str(ep_dir))
    return found, missing


def _count_captions(caption_dir: Path) -> int:
    if not caption_dir.exists():
        return 0
    return len(list(caption_dir.glob("*.json")))


def _count_audits(audit_dir: Path) -> int:
    if not audit_dir.exists():
        return 0
    return len(list(audit_dir.glob("*.json")))


def _check_metrics(summary_path: Path, golden_path: Path) -> list[str]:
    errors: list[str] = []
    summary = json.loads(summary_path.read_text())
    golden = json.loads(golden_path.read_text())

    for modality, g in golden.items():
        if modality not in summary.get("per_modality_confusion", {}):
            errors.append(f"missing modality in summary: {modality}")
            continue
        s = summary["per_modality_confusion"][modality]
        for metric in ("f1", "specificity", "balanced_accuracy"):
            sv = s.get(metric)
            gv = g.get(metric)
            if sv is None or gv is None:
                continue
            if abs(sv - gv) > METRIC_TOLERANCE:
                errors.append(
                    f"{modality}.{metric}: got {sv:.3f}, golden {gv:.3f} "
                    f"(tol ±{METRIC_TOLERANCE})"
                )
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout-dir", type=Path, default=Path("data/rollouts"))
    p.add_argument("--caption-dir", type=Path, default=Path("data/frame_descriptions"))
    p.add_argument("--audit-dir", type=Path, default=Path("data/audits_modality"))
    p.add_argument("--summary", type=Path, default=Path("data/analysis/modality_summary.json"))
    p.add_argument("--golden", type=Path, default=Path("benchmarks/golden_metrics.json"))
    p.add_argument(
        "--check-metrics",
        action="store_true",
        help="Compare modality_summary.json against benchmarks/golden_metrics.json.",
    )
    args = p.parse_args()

    ok = True
    expected_rollouts = len(EXPECTED_TASKS) * len(EXPECTED_VARIANTS) * EXPECTED_EPISODES

    n_rollouts, missing_rollouts = _count_rollouts(args.rollout_dir)
    n_captions = _count_captions(args.caption_dir)
    n_audits = _count_audits(args.audit_dir)

    print(f"Rollouts:  {n_rollouts}/{expected_rollouts} episodes")
    print(f"Captions:  {n_captions}/{expected_rollouts}")
    print(f"Audits:    {n_audits}/{EXPECTED_AUDITS}")

    if n_rollouts < expected_rollouts:
        ok = False
        print(f"  missing {expected_rollouts - n_rollouts} rollout(s)", file=sys.stderr)
        for path in missing_rollouts[:5]:
            print(f"    - {path}", file=sys.stderr)
        if len(missing_rollouts) > 5:
            print(f"    ... and {len(missing_rollouts) - 5} more", file=sys.stderr)
        if not args.rollout_dir.exists():
            print("  hint: run ./scripts/fetch_data.sh", file=sys.stderr)

    if n_captions < expected_rollouts:
        ok = False
        print(f"  missing {expected_rollouts - n_captions} caption file(s)", file=sys.stderr)

    if n_audits < EXPECTED_AUDITS:
        ok = False
        print(f"  missing {EXPECTED_AUDITS - n_audits} audit file(s)", file=sys.stderr)

    if args.check_metrics:
        if not args.summary.exists():
            print(f"[metrics] summary not found: {args.summary}", file=sys.stderr)
            ok = False
        elif not args.golden.exists():
            print(f"[metrics] golden not found: {args.golden}", file=sys.stderr)
            ok = False
        else:
            metric_errors = _check_metrics(args.summary, args.golden)
            if metric_errors:
                ok = False
                print("[metrics] mismatches:", file=sys.stderr)
                for err in metric_errors:
                    print(f"  - {err}", file=sys.stderr)
            else:
                print("[metrics] golden metrics match within tolerance")

    if ok:
        print("Bundle OK")
        return 0

    print("Bundle INCOMPLETE", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
