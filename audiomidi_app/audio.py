from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_TANH1 = math.tanh(1.0)


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int


def read_audio(
    path: str | Path,
    target_sr: int | None = None,
    mono: bool = True,
    normalize: bool = True,
    normalize_mode: str = "peak",
    preemphasis: bool = False,
) -> AudioBuffer:
    data, sr = sf.read(str(path), always_2d=True)
    if mono:
        data = data.mean(axis=1)
    else:
        data = data.T

    if target_sr is not None and target_sr != sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up=up, down=down).astype(np.float32, copy=False)
        sr = target_sr
    else:
        data = data.astype(np.float32, copy=False)

    if normalize:
        if normalize_mode == "rms":
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > 1e-6:
                target_rms = 0.1
                data = data * (target_rms / rms)
                peak = float(np.max(np.abs(data)))
                if peak > 0.99:
                    data = (np.tanh(data) / _TANH1 * 0.99).astype(np.float32)
                data = data.astype(np.float32, copy=False)
        else:
            peak = np.max(np.abs(data))
            if peak > 1e-6:
                data = data * (0.891 / peak)

    if preemphasis:
        data = np.append(data[0], data[1:] - 0.97 * data[:-1]).astype(np.float32, copy=False)

    return AudioBuffer(samples=data, sample_rate=sr)
