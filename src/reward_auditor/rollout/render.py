"""Video rendering via imageio-ffmpeg."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def render_mp4(
    frames: list[np.ndarray] | np.ndarray,
    path: Path,
    fps: int = 30,
) -> Path:
    """Write a list/stack of HxWx3 uint8 frames to an .mp4 at `path`.

    Returns the path (so callers can chain).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected (T, H, W, C) frames; got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    iio.imwrite(path, arr, fps=fps, codec="libx264", macro_block_size=None)
    return path
