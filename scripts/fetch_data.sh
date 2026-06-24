#!/usr/bin/env bash
# Extract rollout videos + trajectories from the bundled archive.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$ROOT/rollouts_iter0.tar.zst"
DEST="$ROOT/data/rollouts"

if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null || true)" ]]; then
  echo "[fetch] $DEST already populated — skipping extract"
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "[fetch] ERROR: archive not found: $ARCHIVE" >&2
  echo "Place rollouts_iter0.tar.zst in the repo root, or download from GitHub Release." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "[fetch] Extracting $ARCHIVE → $DEST"
if tar --help 2>&1 | grep -q use-compress-program; then
  tar --use-compress-program=zstd -xf "$ARCHIVE" -C "$DEST"
else
  zstd -d -c "$ARCHIVE" | tar -xf - -C "$DEST"
fi
echo "[fetch] Done. $(find "$DEST" -name trajectory.npz | wc -l | tr -d ' ') episodes available."
