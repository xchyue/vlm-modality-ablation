# File Ownership Map

This repository was split from the CSE 190 main project [`190SP-Proj`](https://github.com/bruceyuze/190SP-Proj) and bundles the code and data needed to run **Parts 1–3**.  
Read this table before diving into the code: **what is Part 3 research core vs. upstream infrastructure.**

---

## Overview

| Part | Owner | Role in this repo | Do you edit it? |
|------|-------|-------------------|-----------------|
| **Part 1** | Infrastructure | MuJoCo envs, reward components, rollout collection | Rarely; only when adding tasks |
| **Part 2** | Jason | PPO training, policy checkpoints | When adding v1 seeds / retraining |
| **Part 3** | **Xiaochuan** | **M1–M6 modality ablation, metrics, paper results** | **Primary workspace** |
| **Part 4** | Yisheng | VLM auditor (`auditor/vlm.py`) | When swapping models; logic is sufficient as-is |
| **Part 5** | Zhiyuan | Failure-case analysis | **Not included** in this repo |

Legend: **bold = Part 3 core** · plain = upstream dependency · ~~strikethrough~~ = not included

---

## Part 3 — Core (Modality Ablation)

> Paper tables, M1–M6 definitions, specificity / balanced accuracy live here.

### Code

| Path | Purpose |
|------|---------|
| `src/reward_auditor/eval/modality_ablation.py` | **M1–M6 definitions, `build_audit_input` masking** |
| `src/reward_auditor/eval/metrics.py` | **Confusion matrix, F1, specificity, balanced accuracy** |
| `src/reward_auditor/tasks.py` | `TASK_GOALS` extracted from main repo (avoids Part 4 loop dep) |
| `scripts/run_modality_sweep.py` | **Run M1–M6 VLM audits on rollouts** |
| `scripts/analyze_modality_results.py` | **Aggregate 360 audits → tables and plots** |
| `scripts/verify_bundle.py` | Check data completeness + golden metrics |
| `scripts/generate_frame_descriptions.py` | Frame captions for M5 (Gemini API) |
| `tests/test_modality_masking.py` | **M5 caption regression + masking unit tests** |

### Data

| Path | Purpose |
|------|---------|
| `data/audits_modality/` | **360 golden audit JSONs (source of paper numbers)** |
| `data/frame_descriptions/` | Pre-computed M5 captions (60 files) |
| `data/analysis/modality_summary.json` | Golden summary from analyze |
| `benchmarks/golden_metrics.json` | Core metric reference (`verify --check-metrics`) |
| `docs/RESULTS.md` | **Part 3 results write-up** |

### When extending v1 clean runs, change mainly here

1. Add new `v1_ground_truth` episodes under `data/rollouts/` (see Part 1 collection below)
2. Optionally add `data/frame_descriptions/` (for M5)
3. Re-run `run_modality_sweep.py` → new `data/audits_modality/`
4. Update `benchmarks/golden_metrics.json` and `docs/RESULTS.md`
5. Update `EXPECTED_EPISODES` in `scripts/verify_bundle.py` if episodes per variant ≠ 3

---

## Part 1 — Environment & Rollout Collection (upstream)

> Run agents in MuJoCo and write `(video, trajectory, meta)` to disk.

### Code

| Path | Purpose |
|------|---------|
| `src/reward_auditor/envs/` | `make_env`, reward components, `ComponentRewardWrapper` |
| `src/reward_auditor/rollout/collect.py` | **Collect rollouts → npz + mp4 + meta.json** |
| `src/reward_auditor/rollout/render.py` | Write frame sequence to mp4 |
| `src/reward_auditor/rollout/bundle.py` | `RolloutBundle` data structure |
| `src/reward_auditor/variants/` | Load YAML variants, resolve reward weights |
| `configs/tasks/*.yaml` | Per-task default reward weights |
| `configs/variants/*.yaml` | v1–v5 reward variant definitions |
| `scripts/collect_rollouts.py` | **CLI: policy → rollout directory** |
| `scripts/make_env_smoketest.py` | Verify MuJoCo env runs |
| `docs/schemas.md` | On-disk rollout / audit format contract |
| `tests/test_rollout.py` | Rollout pipeline tests |
| `tests/test_components.py` | Reward component unit tests |
| `tests/test_reward_wrapper.py` | Wrapper unit tests |
| `tests/test_variants.py` | YAML loader unit tests |
| `tests/conftest.py` | pytest fixtures |

### Standard command to collect more v1 rollouts

```bash
uv sync --extra train --extra dev

uv run python scripts/collect_rollouts.py \
  --task ant \
  --variant v1_ground_truth \
  --policy data/policies/ant_v1_ground_truth_seed0.pt \
  --n-episodes 5 \
  --out-dir data/rollouts/ant_v1_ground_truth_seed0/iter_0
```

Output layout: `episode_{N}/{trajectory.npz, meta.json, video.mp4}`.  
Part 3 sweep reads `available_components` directly from `meta.json`.

---

## Part 2 — Policy Training (Jason)

> Produces `.pt` checkpoints. Part 3 audits **rollouts**, not checkpoints—but new v1 policies require Part 2.

### Code

| Path | Purpose |
|------|---------|
| `src/reward_auditor/policy/ppo.py` | **PPO training + `load_policy`** |
| `src/reward_auditor/policy/__init__.py` | `PolicyProtocol`, `train`, `load_policy` |
| `scripts/eval_policy.py` | Evaluate hacking severity (v1 vs variant return) |

### Data

| Path | Purpose |
|------|---------|
| `data/policies/*.pt` | **20 pre-trained checkpoints** (4 tasks × 5 variants × seed0) |
| `data/policies/*_metrics.json` | Training / eval metric records |

### Train a new v1 policy (new seed)

```bash
uv run python -m reward_auditor.policy.ppo \
  --task ant \
  --variant v1_ground_truth \
  --seed 1 \
  --out-dir data/policies
```

Then use Part 1 `collect_rollouts.py` to collect rollouts, then run the Part 3 sweep.

---

## Part 4 — VLM Auditor (Yisheng)

> This repo includes **only the audit subset**—not closed-loop reweight training.

### Included

| Path | Purpose |
|------|---------|
| `src/reward_auditor/auditor/vlm.py` | **Gemini 2.5 Flash REST calls, video clipping** |
| `src/reward_auditor/auditor/schemas.py` | `AuditInput` / `AuditOutput` contract |
| `src/reward_auditor/auditor/base.py` | `AuditorProtocol` |
| `src/reward_auditor/auditor/dummy.py` | Dummy auditor for unit tests |

### Not included (still in main repo)

| Path | Reason |
|------|--------|
| `src/reward_auditor/loop/reweight_loop.py` | Part 4 closed-loop retrain; not needed to extend v1 |
| `scripts/run_reweight_loop.py` | Same |
| `scripts/sweep_reweight.py` | Part 4 batch sweep |

For **audit → reweight → retrain PPO** closed loop, use the main `190SP-Proj` repo.

---

## Data directories

| Directory | Part | Size | Git strategy |
|-----------|------|------|--------------|
| `data/audits_modality/` | **3** | ~12 MB | commit |
| `data/frame_descriptions/` | **3** | ~240 KB | commit |
| `data/analysis/` | **3** | ~280 KB | commit |
| `data/rollouts/` | 1 → **3** consumes | ~93 MB | tarball / Release |
| `data/policies/` | 2 | ~10 MB | commit (`.gitignore` exception) |
| `rollouts_iter0.tar.zst` | 1 | ~87 MB | Release or optional commit |

---

## Install commands by goal

| Goal | Command |
|------|---------|
| Reproduce Part 3 tables only (L0) | `uv sync --extra audit` |
| Re-run VLM sweep (L1) | above + `GEMINI_API_KEY` + `./scripts/fetch_data.sh` |
| Collect rollouts / train policies (L2) | `uv sync --extra train --extra dev` |
| Everything | `uv sync --extra all` |

On headless servers for train/collect: `export MUJOCO_GL=egl`.

---

## Recommended workflow: add more v1 clean runs

```text
Part 2  train/load policy (.pt)
    ↓
Part 1  collect_rollouts → data/rollouts/.../episode_*
    ↓
Part 3  generate_frame_descriptions (optional, for M5)
    ↓
Part 3  run_modality_sweep → data/audits_modality/
    ↓
Part 3  analyze_modality_results + verify_bundle --check-metrics
```

---

## Relationship to the main repo

- **Canonical Part 3 benchmark** = this repo
- **Canonical Part 4 closed loop** = main `190SP-Proj` repo
- The two repos **do not auto-sync**; cherry-pick or subtree manually after large changes

For ownership questions, use this file; for on-disk formats see `docs/schemas.md`.
