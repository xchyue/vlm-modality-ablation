"""Run modality ablation sweep on bundled rollouts.

For each episode in data/rollouts/, call VLMAuditor under M1–M6 modalities
(with appropriate signal masking) and save the audit JSON per modality.

Usage:
    GEMINI_API_KEY=... uv run python scripts/run_modality_sweep.py \\
        --modalities M1 M2 M3 \\
        --tasks ant halfcheetah hopper humanoid \\
        --max-episodes-per-variant 3

    # Smoke test: 1 episode, just M1 (cheapest)
    GEMINI_API_KEY=... uv run python scripts/run_modality_sweep.py \\
        --modalities M1 --max-episodes-per-variant 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from reward_auditor.auditor import VLMAuditor, VLMClientConfig
from reward_auditor.eval.modality_ablation import (
    MODALITY_CONFIGS,
    Modality,
    build_audit_input,
)
from reward_auditor.rollout import RolloutBundle
from reward_auditor.tasks import (
    _safe_filename,
    available_components_from_meta,
    default_task_goal,
)


def _load_frame_descriptions(
    frame_desc_dir: Path,
    task: str,
    variant: str,
    episode_idx: int,
) -> list[str] | None:
    """Load pre-computed frame captions for an episode, if available."""
    cache = frame_desc_dir / f"{task}_{variant}_episode{episode_idx}.json"
    if not cache.exists():
        return None
    return json.loads(cache.read_text())["frames"]


def _load_bundle(ep_dir: Path) -> RolloutBundle:
    """Rebuild a RolloutBundle from on-disk rollout layout."""
    npz = np.load(ep_dir / "trajectory.npz")
    component_log = {
        key[len("component_") :]: np.asarray(npz[key])
        for key in npz.files
        if key.startswith("component_")
    }
    meta = json.loads((ep_dir / "meta.json").read_text())
    rewards = np.asarray(npz["rewards"])
    return RolloutBundle(
        task=meta["task"],
        variant=meta["variant"],
        weights=meta["weights"],
        video_path=ep_dir / "video.mp4",
        states=np.asarray(npz["states"]),
        actions=np.asarray(npz["actions"]),
        rewards=rewards,
        component_log=component_log,
        episode_return=float(meta.get("episode_return", float(rewards.sum()))),
        episode_length=int(meta.get("episode_length", len(rewards))),
        metadata=meta,
    )


def _discover_episodes(
    rollout_root: Path,
    tasks: list[str],
    iter_idx: int,
    max_episodes_per_variant: int,
) -> list[tuple[str, str, int, Path]]:
    """Find (task, variant, episode_idx, episode_dir) tuples to audit."""
    out = []
    for variant_dir in sorted(rollout_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        name = variant_dir.name
        task_match = next((t for t in tasks if name.startswith(t + "_")), None)
        if task_match is None:
            continue
        variant = name[len(task_match) + 1 :].rsplit("_seed", 1)[0]
        iter_dir = variant_dir / f"iter_{iter_idx}"
        if not iter_dir.exists():
            continue
        for ep_idx in range(max_episodes_per_variant):
            ep_dir = iter_dir / f"episode_{ep_idx}"
            if (ep_dir / "trajectory.npz").exists():
                out.append((task_match, variant, ep_idx, ep_dir))
    return out


def _make_auditor(needs_video: bool) -> VLMAuditor:
    return VLMAuditor(
        VLMClientConfig(
            send_video=needs_video,
            require_video=needs_video,
        )
    )


def _output_path(
    audit_dir: Path,
    task: str,
    variant: str,
    episode_idx: int,
    modality: Modality,
    auditor_name: str,
) -> Path:
    safe_name = _safe_filename(auditor_name)
    return audit_dir / (
        f"{task}_{variant}_episode{episode_idx}"
        f"_{modality.value}"
        f"_{safe_name}.json"
    )


_ALL_MODALITIES = [m.value for m in Modality]


def main() -> int:
    p = argparse.ArgumentParser(description="Run modality ablation sweep.")
    p.add_argument(
        "--modalities",
        nargs="+",
        choices=_ALL_MODALITIES,
        default=["M1", "M2", "M3"],
        help="Which modalities to audit (default M1 M2 M3 — cheap, no video).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["ant", "halfcheetah", "hopper", "humanoid"],
        help="Which tasks to include.",
    )
    p.add_argument(
        "--iter-idx",
        type=int,
        default=0,
        help="Which iter_N subdir to audit (default 0 = initial policy).",
    )
    p.add_argument(
        "--max-episodes-per-variant",
        type=int,
        default=3,
        help="Max episodes per (task, variant) to audit (default 3).",
    )
    p.add_argument("--rollout-dir", type=Path, default=Path("data/rollouts"))
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("data/audits_modality"),
        help="Output dir for per-modality audits.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep-between-calls", type=float, default=1.0)
    p.add_argument(
        "--frame-desc-dir",
        type=Path,
        default=Path("data/frame_descriptions"),
        help="Pre-computed frame captions (for M5).",
    )
    args = p.parse_args()

    if not args.rollout_dir.exists():
        print(
            f"[sweep] rollout dir not found: {args.rollout_dir}\n"
            "Run ./scripts/fetch_data.sh to extract rollouts_iter0.tar.zst first.",
            file=sys.stderr,
        )
        return 1

    args.audit_dir.mkdir(parents=True, exist_ok=True)

    episodes = _discover_episodes(
        rollout_root=args.rollout_dir,
        tasks=args.tasks,
        iter_idx=args.iter_idx,
        max_episodes_per_variant=args.max_episodes_per_variant,
    )
    modalities = [Modality(m) for m in args.modalities]

    print(
        f"[sweep] Found {len(episodes)} episodes across "
        f"{len({(t, v) for t, v, _, _ in episodes})} (task, variant) pairs"
    )
    print(f"[sweep] Will audit {len(episodes) * len(modalities)} (episode × modality) pairs")
    print(f"[sweep] Output dir: {args.audit_dir}")

    if args.dry_run:
        print("[sweep] DRY RUN — listing work:")
        for task, variant, ep_idx, _ in episodes:
            for m in modalities:
                print(f"  {task} {variant} ep{ep_idx} {m.value}")
        return 0

    video_modalities = {m for m in modalities if MODALITY_CONFIGS[m].include_video}
    nonvideo_modalities = set(modalities) - video_modalities
    auditor_with_video = _make_auditor(needs_video=True) if video_modalities else None
    auditor_no_video = _make_auditor(needs_video=False) if nonvideo_modalities else None

    done = 0
    skipped = 0
    failed = 0
    for task, variant, ep_idx, ep_dir in episodes:
        try:
            bundle = _load_bundle(ep_dir)
        except Exception as e:
            print(f"[sweep] ✗ load failed {ep_dir}: {e}", file=sys.stderr)
            failed += 1
            continue

        for modality in modalities:
            cfg = MODALITY_CONFIGS[modality]
            auditor = auditor_with_video if cfg.include_video else auditor_no_video
            assert auditor is not None

            out_path = _output_path(
                args.audit_dir, task, variant, ep_idx, modality, auditor.name
            )
            if out_path.exists() and not args.overwrite:
                print(f"  skip (exists): {out_path.name}")
                skipped += 1
                continue

            frame_desc = None
            if MODALITY_CONFIGS[modality].include_frame_descriptions:
                frame_desc = _load_frame_descriptions(
                    args.frame_desc_dir, task, variant, ep_idx
                )
                if frame_desc is None:
                    print(
                        f"  skip (no captions): {out_path.name} — "
                        f"run generate_frame_descriptions.py first",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue

            audit_input = build_audit_input(
                bundle=bundle,
                task_goal=default_task_goal(task),
                available_components=available_components_from_meta(bundle.metadata),
                modality=modality,
                frame_descriptions=frame_desc,
            )

            try:
                t0 = time.time()
                audit_output = auditor.audit(audit_input)
                elapsed = time.time() - t0

                record = {
                    "audit_input": audit_input.model_dump(mode="json"),
                    "audit_output": audit_output.model_dump(mode="json"),
                    "auditor_name": f"{auditor.name}[{modality.value}]",
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                }
                out_path.write_text(json.dumps(record, indent=2))
                print(
                    f"  ✓ {out_path.name}  "
                    f"hacking={audit_output.reward_hacking_detected} "
                    f"({elapsed:.1f}s)"
                )
                done += 1
            except Exception as e:
                print(f"  ✗ {out_path.name}: {e}", file=sys.stderr)
                failed += 1

            if args.sleep_between_calls > 0:
                time.sleep(args.sleep_between_calls)

    print()
    print(f"[sweep] done={done} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
