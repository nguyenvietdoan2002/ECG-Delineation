"""Synthetic 12-lead ECGs with known wave positions, for tests (ported from the sibling ecgrepro project).

Each beat is a sum of Gaussian P, Q, R, S and T waves at fixed offsets from its R peak. Leads scale
the waves differently: aVR and aVL are inverted and V1 has an rS shape with an inverted T.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FS = 500
LEAD_GAINS = np.array([1.0, 1.2, 0.4, -0.9, -0.3, 0.8, 1.0, 1.1, 1.3, 1.2, 1.0, 0.8])
V1 = 6
V1_WAVES = {"P": 0.08, "Q": -0.10, "R": 0.35, "S": -1.10, "T": -0.20}
# wave: (offset from R in s, amplitude in mV, Gaussian width in s)
WAVES = {
    "P": (-0.16, 0.15, 0.020),
    "Q": (-0.03, -0.10, 0.008),
    "R": (0.00, 1.00, 0.010),
    "S": (0.03, -0.25, 0.008),
    "T": (0.28, 0.30, 0.045),
}


@dataclass(frozen=True)
class Synthetic:
    signal: NDArray[np.float64]  # (12, samples), mV
    fs: int
    waves: dict[str, NDArray[np.int64]]  # wave name -> sample index of its peak in each beat


def synthetic_ecg(
    rr: float | Sequence[float] = 1.0,
    duration: float = 10.0,
    fs: int = FS,
    p_waves: bool = True,
    t_sign: float = 1.0,
    noise_mv: float = 0.0,
    seed: int = 0,
) -> Synthetic:
    """A clean ECG. ``rr`` is one RR interval (regular rhythm) or a list cycled through."""
    intervals = [rr] if isinstance(rr, float | int) else list(rr)
    r_times = [0.6]
    while r_times[-1] + intervals[len(r_times) % len(intervals)] < duration - 0.6:
        r_times.append(r_times[-1] + intervals[len(r_times) % len(intervals)])
    r = np.array(r_times)
    t = np.arange(int(duration * fs)) / fs
    signal = np.zeros((12, t.size))
    for name, (offset, amplitude, width) in WAVES.items():
        if name == "P" and not p_waves:
            continue
        shape = sum(np.exp(-0.5 * ((t - centre) / width) ** 2) for centre in r + offset)
        amplitudes = LEAD_GAINS * amplitude
        amplitudes[V1] = V1_WAVES[name]
        signal += (t_sign if name == "T" else 1.0) * amplitudes[:, None] * shape[None, :]
    if noise_mv:
        signal += np.random.default_rng(seed).normal(0.0, noise_mv, signal.shape)
    waves = {name: np.round((r + offset) * fs).astype(np.int64) for name, (offset, _, _) in WAVES.items()}
    return Synthetic(signal=signal, fs=fs, waves=waves)
