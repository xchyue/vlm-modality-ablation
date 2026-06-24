"""Generate frame captions for each episode video using Gemini Flash.

For M5 ablation modality, we need pre-computed frame descriptions on top of
video. This script clips each rollout video to ~5s, asks Gemini to caption it
in N sentences, and caches the result.

Usage:
    GEMINI_API_KEY=... uv run python scripts/generate_frame_descriptions.py \\
        --tasks ant halfcheetah \\
        --max-episodes-per-variant 3

    # Smoke test: 1 video
    GEMINI_API_KEY=... uv run python scripts/generate_frame_descriptions.py \\
        --tasks ant --max-episodes-per-variant 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reward_auditor.auditor.vlm import (
    VLMAuditor,
    VLMClientConfig,
    _video_inline_part,  # pylint: disable=protected-access
)

N_FRAME_CAPTIONS = 8


def _build_caption_prompt(n_captions: int) -> str:
    return (
        f"Watch this short clip of a MuJoCo locomotion policy. Describe what "
        f"the agent is doing in exactly {n_captions} short sentences, one per "
        f"approximately equal time interval. Focus on observable physical "
        f"behavior: posture, gait, contact with ground, whether motion is "
        f"smooth or chaotic, whether the agent appears to be performing the "
        f"task or doing something abnormal.\n\n"
        f"Return a JSON object with a single key 'frames' whose value is a "
        f"list of exactly {n_captions} strings. Do not include any commentary "
        f"outside the JSON."
    )


def _caption_one_video(
    auditor: VLMAuditor,
    video_path: Path,
    n_captions: int,
) -> list[str]:
    video_part = _video_inline_part(
        video_path,
        max_seconds=auditor.config.max_video_seconds,
        sampling_fps=auditor.config.video_fps,
    )
    if video_part is None:
        raise RuntimeError(f"Could not clip video: {video_path}")

    system = "You are an embodied behavior describer. Output strict JSON only."
    contents = [
        {
            "role": "user",
            "parts": [
                video_part,
                {"text": _build_caption_prompt(n_captions)},
            ],
        }
    ]
    response = auditor._api_call(system, contents)  # pylint: disable=protected-access
    text = response["candidates"][0]["content"]["parts"][0]["text"]

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    parsed = json.loads(stripped.strip())
    frames = parsed["frames"]
    if not isinstance(frames, list) or not all(isinstance(s, str) for s in frames):
        raise ValueError(f"Unexpected frames format: {frames!r}")
    return frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["ant", "halfcheetah", "hopper", "humanoid"],
    )
    p.add_argument("--iter-idx", type=int, default=0)
    p.add_argument("--max-episodes-per-variant", type=int, default=3)
    p.add_argument("--rollout-dir", type=Path, default=Path("data/rollouts"))
    p.add_argument("--out-dir", type=Path, default=Path("data/frame_descriptions"))
    p.add_argument("--n-captions", type=int, default=N_FRAME_CAPTIONS)
    p.add_argument("--sleep-between-calls", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.rollout_dir.exists():
        print(
            f"[caption] rollout dir not found: {args.rollout_dir}\n"
            "Run ./scripts/fetch_data.sh to extract rollouts_iter0.tar.zst first.",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    work = []
    for variant_dir in sorted(args.rollout_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        name = variant_dir.name
        task = next((t for t in args.tasks if name.startswith(t + "_")), None)
        if task is None:
            continue
        variant = name[len(task) + 1 :].rsplit("_seed", 1)[0]
        iter_dir = variant_dir / f"iter_{args.iter_idx}"
        if not iter_dir.exists():
            continue
        for ep_idx in range(args.max_episodes_per_variant):
            video = iter_dir / f"episode_{ep_idx}" / "video.mp4"
            if video.exists():
                work.append((task, variant, ep_idx, video))

    print(f"[caption] Found {len(work)} videos to caption")
    if args.dry_run:
        for task, variant, ep_idx, video in work:
            print(f"  {task} {variant} ep{ep_idx} → {video}")
        return 0

    auditor = VLMAuditor(VLMClientConfig(send_video=True, require_video=True))

    done = skipped = failed = 0
    for task, variant, ep_idx, video in work:
        out_path = args.out_dir / f"{task}_{variant}_episode{ep_idx}.json"
        if out_path.exists() and not args.overwrite:
            print(f"  skip: {out_path.name}")
            skipped += 1
            continue

        try:
            t0 = time.time()
            frames = _caption_one_video(auditor, video, args.n_captions)
            elapsed = time.time() - t0
            out_path.write_text(
                json.dumps(
                    {
                        "task": task,
                        "variant": variant,
                        "episode": ep_idx,
                        "video": str(video),
                        "frames": frames,
                    },
                    indent=2,
                )
            )
            print(f"  ✓ {out_path.name}  {len(frames)} captions  ({elapsed:.1f}s)")
            done += 1
        except Exception as e:
            print(f"  ✗ {out_path.name}: {e}", file=sys.stderr)
            failed += 1

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    print()
    print(f"[caption] done={done} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
