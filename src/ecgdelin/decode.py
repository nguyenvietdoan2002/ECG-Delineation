"""Per-sample class regions -> NeuroKit2-style points.

Clean-up fixed in advance (decision 27): same-wave pieces separated by a gap under 10 ms are joined,
then P and QRS regions under 20 ms and T regions under 40 ms are dropped. Onset and offset are the
region's first and last sample; the P and T peak is the largest absolute deflection from the region's
baseline (the mean of its two boundary samples); the QRS peak is the highest positive deflection when
one exists, as LUDB marks the R wave, else the nadir (decision 47); Q and S are the lowest points
before and after the R peak inside the QRS region. Regions are grouped into beats as in
`delineate.group_beats`.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import Wave
from ecgdelin.delineate import LeadPoints, group_beats, points_from_beats
from ecgdelin.labels import CLASSES

MIN_MS = {"P": 20, "QRS": 20, "T": 40}
JOIN_GAP_MS = 10


def runs(classes: NDArray[np.int8]) -> list[tuple[str, int, int]]:
    """Maximal runs of one non-background class as (kind, first sample, last sample)."""
    out = []
    n = classes.size
    i = 0
    while i < n:
        c = int(classes[i])
        j = i
        while j + 1 < n and classes[j + 1] == c:
            j += 1
        if c != 0:
            out.append((CLASSES[c], i, j))
        i = j + 1
    return out


def clean(regions: Sequence[tuple[str, int, int]], fs: int) -> list[tuple[str, int, int]]:
    gap = int(round(JOIN_GAP_MS * fs / 1000))
    joined: list[tuple[str, int, int]] = []
    for kind, start, end in regions:
        if joined and joined[-1][0] == kind and start - joined[-1][2] - 1 < gap:
            joined[-1] = (kind, joined[-1][1], end)
        else:
            joined.append((kind, start, end))
    return [(k, s, e) for k, s, e in joined if e - s + 1 >= int(round(MIN_MS[k] * fs / 1000))]


MIN_R_MV = 0.02  # a positive QRS deflection at least this high counts as an R wave (decision 47)


def _peak(kind: str, signal: NDArray[np.float64], start: int, end: int) -> int:
    deflection = signal[start : end + 1] - (signal[start] + signal[end]) / 2
    if kind == "QRS":
        # LUDB marks the R wave even when the S wave is deeper; only a QS complex gets its nadir.
        return start + int(np.argmax(deflection) if deflection.max() > MIN_R_MV else np.argmin(deflection))
    return start + int(np.argmax(np.abs(deflection)))


def waves_from_classes(classes: NDArray[np.int8], signal: NDArray[np.float64], fs: int) -> list[Wave]:
    """One lead's cleaned regions as waves with their peaks, orphans included."""
    return [Wave(kind, start, _peak(kind, signal, start, end), end) for kind, start, end in clean(runs(classes), fs)]


def regions_to_points(classes: NDArray[np.int8], signal: NDArray[np.float64], fs: int) -> LeadPoints:
    """One lead: ``classes`` (indices into CLASSES) and the lead's signal, both of the same length."""
    waves = waves_from_classes(classes, signal, fs)
    beats, orphans = group_beats(waves)
    q_peaks = []
    s_peaks = []
    for beat in beats:
        onset, r, offset = beat.qrs.onset, beat.qrs.peak, beat.qrs.offset
        assert onset is not None and offset is not None
        q_peaks.append(onset + int(np.argmin(signal[onset:r])) if r > onset else math.nan)
        s_peaks.append(r + 1 + int(np.argmin(signal[r + 1 : offset + 1])) if r < offset else math.nan)
    return points_from_beats(beats, q_peaks, s_peaks, orphans)
