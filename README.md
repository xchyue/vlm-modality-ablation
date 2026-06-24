# VLM Modality Ablation Benchmark

Standalone benchmark for **Part 3** of the CSE 190 reward-hacking auditor project:
a VLM (Gemini 2.5 Flash) judges whether RL policies were trained under clean or hacked
rewards, while varying **which signals the auditor sees** across six modalities (M1–M6).

**Coverage:** 4 tasks × 5 reward variants × 6 modalities × 3 episodes = **360 audits**.

Also bundles **Part 1 + Part 2** code to train policies and collect new rollouts (e.g. more
`v1_ground_truth` clean runs). See **[docs/PART_FILES.md](docs/PART_FILES.md)** for which
files belong to which part.

## Quick start

```bash
# Audit-only (no MuJoCo / torch)
uv sync --extra audit

# Full stack: train + collect + audit
uv sync --extra all

# Extract rollout data (~92 MB compressed archive)
chmod +x scripts/fetch_data.sh
./scripts/fetch_data.sh

# Verify bundle
uv run python scripts/verify_bundle.py

# Reproduce paper numbers (no API key)
uv run python scripts/analyze_modality_results.py
uv run python scripts/verify_bundle.py --check-metrics
```

## Reproduction levels

| Level | What you can do | Requires |
|-------|-----------------|----------|
| **L0** | Verify tables in `docs/RESULTS.md` | `data/audits_modality/` + analyze |
| **L1** | Re-run M1–M6 with your API key | L0 + rollouts + `GEMINI_API_KEY` |
| **L2** | Train policy + collect new v1 rollouts | `--extra train` + `data/policies/` |
| **L3** | Extend modalities / metrics | L1 + `src/reward_auditor/eval/` |

See [REPRODUCE.md](REPRODUCE.md) and [docs/PART_FILES.md](docs/PART_FILES.md).

## Key results (golden)

| Mod | Signal | F1 | Balanced Acc |
|-----|--------|-----|--------------|
| M1 | goal only | 0.000 | 0.500 |
| M2 | + reward code | **0.845** | 0.594 |
| M5 | all signals | 0.868 | 0.479 |
| M6 | video only | 0.738 | **0.615** |

Full write-up: [docs/RESULTS.md](docs/RESULTS.md).

## Layout

```
docs/PART_FILES.md     ← Part 1/2/3/4 ownership map (read first)
configs/               Part 1 — task & variant YAML
src/reward_auditor/
  envs/ policy/        Part 1 + 2 — train & collect
  eval/ tasks.py       Part 3 — modality ablation (core)
  auditor/             Part 4 subset — VLM only
scripts/               sweep / analyze / collect / train entrypoints
data/
  policies/            Part 2 — 20 checkpoints
  rollouts/            Part 1 output → Part 3 input
  audits_modality/     Part 3 golden audits
```

## License

MIT — see [LICENSE](LICENSE).
