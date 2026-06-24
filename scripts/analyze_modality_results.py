"""Analyze modality ablation results: load all audit JSONs, compute metrics,
generate plots for Slide 6.

DROP AT: scripts/analyze_modality_results.py

Reads:
  data/audits_modality/*.json        (user's M1-M4 ablation sweep)
  data/audits/*.json                 (optional: team's single-modality, treated as M5_no_frame_desc)

Outputs:
  data/analysis/modality_summary.json     - machine-readable summary
  data/analysis/modality_report.txt       - human-readable report
  data/analysis/plots/                    - matplotlib figures (PNG)

Usage:
    uv run python scripts/analyze_modality_results.py

    # Include team's data/audits/ as M5_no_frame_desc baseline
    uv run python scripts/analyze_modality_results.py --include-team-audits

    # Skip plots (text-only mode, no matplotlib)
    uv run python scripts/analyze_modality_results.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from reward_auditor.eval.metrics import ConfusionMatrix, is_hacking_variant

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(
    r"^(?P<task>ant|halfcheetah|hopper|humanoid)"
    r"_(?P<variant>v[1-5]_[a-z_]+?)"
    r"_episode(?P<ep>\d+)"
    r"(?:_(?P<modality>M[1-6]))?"
    r"_VLMAuditor.*\.json$"
)


def parse_filename(name: str) -> dict | None:
    """Parse audit filename → metadata dict."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "task": m.group("task"),
        "variant": m.group("variant"),
        "episode": int(m.group("ep")),
        "modality": m.group("modality") or "M5_no_fd",
    }

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audits(dirs: list[Path]) -> list[dict[str, Any]]:
    records = []
    for d in dirs:
        if not d.exists():
            print(f"[warn] dir not found: {d}", file=sys.stderr)
            continue
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("v0"):
                continue
            if "direct_baseline" in f.name:
                continue
            meta = parse_filename(f.name)
            if meta is None:
                continue
            try:
                data = json.loads(f.read_text())
                ao = data["audit_output"]
                records.append({
                    **meta,
                    "pred_hacking": ao["reward_hacking_detected"],
                    "task_success": ao["task_success"],
                    "severity": ao.get("severity"),
                    "reason": ao.get("reason", ""),
                    "filename": f.name,
                })
            except Exception as e:
                print(f"[skip] {f.name}: {e}", file=sys.stderr)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def confusion_per_modality(records: list[dict]) -> dict[str, ConfusionMatrix]:
    out: dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    for r in records:
        cm = out[r["modality"]]
        true_h = is_hacking_variant(r["variant"])
        pred = r["pred_hacking"]
        if true_h and pred:        cm.tp += 1
        elif true_h:               cm.fn += 1
        elif pred:                 cm.fp += 1
        else:                      cm.tn += 1
    return dict(out)


def detection_rate_per_cell(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{modality: {task: {variant: detection_rate}}}"""
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in records:
        c = counts[r["modality"]][r["task"]][r["variant"]]
        c[0] += int(r["pred_hacking"])  # n_detected
        c[1] += 1                       # n_total
    out: dict = {}
    for m in counts:
        out[m] = {}
        for t in counts[m]:
            out[m][t] = {
                v: counts[m][t][v][0] / counts[m][t][v][1]
                for v in counts[m][t]
            }
    return out


def accuracy_per_cell(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{modality: {task: {variant: accuracy}}}"""
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in records:
        true_h = is_hacking_variant(r["variant"])
        c = counts[r["modality"]][r["task"]][r["variant"]]
        c[0] += int(r["pred_hacking"] == true_h)  # n_correct
        c[1] += 1                                  # n_total
    out: dict = {}
    for m in counts:
        out[m] = {}
        for t in counts[m]:
            out[m][t] = {
                v: counts[m][t][v][0] / counts[m][t][v][1]
                for v in counts[m][t]
            }
    return out


def failure_cases(records: list[dict], modality: str) -> list[dict]:
    """Audits where VLM was wrong (FN or FP) for given modality."""
    out = []
    for r in records:
        if r["modality"] != modality:
            continue
        true_h = is_hacking_variant(r["variant"])
        if r["pred_hacking"] != true_h:
            out.append({
                "filename": r["filename"],
                "task": r["task"],
                "variant": r["variant"],
                "episode": r["episode"],
                "error_type": "FN" if true_h else "FP",
                "reason": r["reason"][:300],
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def render_text_report(
    records: list[dict],
     cms: dict[str, ConfusionMatrix],
    detect_rates: dict[str, dict[str, dict[str, float]]],
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"MODALITY ABLATION REPORT  ({len(records)} audits)")
    lines.append("=" * 72)
    lines.append("")

    # Per-modality confusion
    lines.append("Per-modality confusion matrix:")
    lines.append("")
    lines.append(f"{'Mod':<8} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5} "
                 f"{'Acc':<7} {'Prec':<7} {'Rec':<7} {'F1':<7} {'Spec':<7} {'BalAcc':<7}")
    lines.append("-" * 76)
    for mod in sorted(cms):
        cm = cms[mod]
        lines.append(
            f"{mod:<8} {cm.tp:<5} {cm.fp:<5} {cm.tn:<5} {cm.fn:<5} "
            f"{cm.accuracy:<7.3f} {cm.precision:<7.3f} {cm.recall:<7.3f} {cm.f1:<7.3f} "
            f"{cm.specificity:<7.3f} {cm.balanced_accuracy:<7.3f}"
        )
    lines.append("")

    # Coverage
    lines.append("=" * 72)
    lines.append("Coverage: (task, variant, modality) cells")
    lines.append("=" * 72)
    cov = defaultdict(lambda: defaultdict(set))
    for r in records:
        cov[r["task"]][r["variant"]].add((r["modality"], r["episode"]))
    for t in sorted(cov):
        lines.append(f"\n{t}:")
        for v in sorted(cov[t]):
            cells = sorted(cov[t][v])
            mods = sorted({c[0] for c in cells})
            eps = sorted({c[1] for c in cells})
            lines.append(f"  {v:<22}  modalities={mods}  episodes={eps}  ({len(cells)} audits)")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (optional, requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(
    records: list[dict],
    cms: dict[str, ConfusionMatrix],
    detect_rates: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed, skipping plots", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    modalities = sorted(cms.keys())

    # Plot 1: Per-modality metric bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modalities))
    width = 0.18
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, metric in enumerate(metrics):
        vals = [getattr(cms[m], metric) for m in modalities]
        ax.bar(x + (i - 1.5) * width, vals, width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Score")
    ax.set_title("Modality ablation: per-modality metrics")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_per_modality.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_dir / 'metrics_per_modality.png'}")

    # Plot 2: Detection rate heatmap (modality × variant)
    all_variants = sorted({r["variant"] for r in records})
    grid = np.full((len(modalities), len(all_variants)), np.nan)
    for i, m in enumerate(modalities):
        for j, v in enumerate(all_variants):
            rates = []
            for t in detect_rates.get(m, {}):
                if v in detect_rates[m][t]:
                    rates.append(detect_rates[m][t][v])
            if rates:
                grid[i, j] = sum(rates) / len(rates)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(all_variants)))
    ax.set_xticklabels(all_variants, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(modalities)))
    ax.set_yticklabels(modalities)
    ax.set_title("Hacking detection rate (modality × variant)")
    for i in range(len(modalities)):
        for j in range(len(all_variants)):
            if not np.isnan(grid[i, j]):
                color = "white" if grid[i, j] < 0.5 else "black"
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="Detection rate")
    fig.tight_layout()
    fig.savefig(out_dir / "detection_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_dir / 'detection_heatmap.png'}")

    # Plot 3: Subtlety curve (V1 → V5 per modality)
    fig, ax = plt.subplots(figsize=(10, 5))
    variant_order = [
        "v1_ground_truth", "v2_subtle", "v3_shaping", "v4_blatant", "v5_sim_bug",
    ]
    for m in modalities:
        ys = []
        for v in variant_order:
            rates = []
            for t in detect_rates.get(m, {}):
                if v in detect_rates[m][t]:
                    rates.append(detect_rates[m][t][v])
            ys.append(sum(rates) / len(rates) if rates else np.nan)
        ax.plot(variant_order, ys, marker="o", label=m, linewidth=2)
    ax.set_ylabel("Detection rate")
    ax.set_xlabel("Variant (subtlety →)")
    ax.set_title("Subtlety curve: detection rate per variant per modality")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "subtlety_curve.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_dir / 'subtlety_curve.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--include-team-audits",
        action="store_true",
        help="Also load data/audits/ (team's data, treated as M5_no_fd).",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for summary + plots.",
    )
    args = p.parse_args()

    # Load
    dirs = [Path("data/audits_modality")]
    if args.include_team_audits:
        dirs.append(Path("data/audits"))
    records = load_audits(dirs)
    print(f"Loaded {len(records)} audit records from {dirs}")
    if not records:
        print("No records loaded. Did you run the modality sweep yet?", file=sys.stderr)
        return 1

    # Analyze
    cms = confusion_per_modality(records)
    detect_rates = detection_rate_per_cell(records)
    acc_per_cell = accuracy_per_cell(records)

    # Text report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = render_text_report(records, cms, detect_rates)
    print()
    print(report)
    (args.output_dir / "modality_report.txt").write_text(report)
    print(f"\n[wrote] {args.output_dir / 'modality_report.txt'}")

    # JSON summary
    summary = {
        "n_audits": len(records),
        "per_modality_confusion": {m: cms[m].as_dict() for m in cms},
        "detection_rate_per_cell": detect_rates,
        "accuracy_per_cell": acc_per_cell,
        "failure_cases": {m: failure_cases(records, m) for m in cms},
    }
    (args.output_dir / "modality_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"[wrote] {args.output_dir / 'modality_summary.json'}")

    # Plots
    if not args.no_plots:
        print()
        print("[plots]")
        make_plots(records, cms, detect_rates, args.output_dir / "plots")

    print()
    print("Done. Open data/analysis/plots/*.png to view figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())