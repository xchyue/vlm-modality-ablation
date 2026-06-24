# Reproduction Guide — VLM Modality Ablation

Which files belong to Part 1 / 2 / 3 / 4? → **[docs/PART_FILES.md](docs/PART_FILES.md)**

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended)
- `zstd` for rollout archive (`brew install zstd` on macOS)
- `GEMINI_API_KEY` — only for re-running VLM audits (L1)
- MuJoCo + torch — only for training / collecting rollouts (L2)

## L0 — Verify published numbers (no API, no MuJoCo)

```bash
cd vlm-modality-ablation
uv sync --extra audit

uv run python scripts/verify_bundle.py
# Expect: 60 captions, 360 audits (rollouts 0/60 until fetch — OK for L0)

uv run python scripts/analyze_modality_results.py
uv run python scripts/verify_bundle.py --check-metrics
```

Expected: M2 F1 ≈ 0.845, M5 specificity = 0.000, M6 balanced acc ≈ 0.615.

## L1 — Full VLM re-run

```bash
./scripts/fetch_data.sh
export GEMINI_API_KEY="your-key"

uv run python scripts/run_modality_sweep.py \
  --modalities M1 M2 M3 M4 M5 M6 \
  --tasks ant halfcheetah hopper humanoid \
  --max-episodes-per-variant 3 \
  --overwrite

uv run python scripts/analyze_modality_results.py
```

Smoke test: `--modalities M1 --tasks ant --max-episodes-per-variant 1`

## L2 — Train policy + collect rollouts (Part 1 + 2)

```bash
uv sync --extra train --extra dev

# Verify MuJoCo env
uv run python scripts/make_env_smoketest.py --task ant --variant v1_ground_truth

# Option A: use bundled checkpoint
uv run python scripts/collect_rollouts.py \
  --task ant --variant v1_ground_truth \
  --policy data/policies/ant_v1_ground_truth_seed0.pt \
  --n-episodes 5 \
  --out-dir data/rollouts/ant_v1_ground_truth_seed0/iter_0

# Option B: train new seed then collect
uv run python -m reward_auditor.policy.ppo \
  --task ant --variant v1_ground_truth --seed 1

uv run python scripts/collect_rollouts.py \
  --task ant --variant v1_ground_truth \
  --policy data/policies/ant_v1_ground_truth_seed1.pt \
  --n-episodes 3 \
  --out-dir data/rollouts/ant_v1_ground_truth_seed1/iter_0
```

Then continue with L1 sweep on the new rollouts. Update `verify_bundle.py` if episode
counts change.

Headless server: `export MUJOCO_GL=egl`

## L3 — Unit tests

```bash
# Part 3 core (no MuJoCo)
uv run pytest tests/test_modality_masking.py -v

# Part 1 fast tests
uv run pytest tests/test_components.py tests/test_reward_wrapper.py tests/test_variants.py -v

# MuJoCo-heavy (optional)
uv run pytest tests/ -m slow -v
```

## Data inventory

| Path | Part | In git? |
|------|------|---------|
| `data/audits_modality/` | 3 | yes |
| `data/frame_descriptions/` | 3 | yes |
| `data/policies/*.pt` | 2 | yes (~10 MB) |
| `data/rollouts/` | 1→3 | tarball |
| `rollouts_iter0.tar.zst` | 1 | Release optional |

## Publishing to GitHub

```bash
git init
git add README.md REPRODUCE.md pyproject.toml LICENSE .gitignore uv.lock \
  configs/ scripts/ src/ tests/ docs/ benchmarks/ data/
git commit -m "Benchmark bundle v0.2.0 with Part 1–3 stack"
```

See `docs/PART_FILES.md` for what to include and what each directory does.
