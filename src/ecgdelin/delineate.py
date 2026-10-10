"""The delineator interface: a 12-lead ECG in, NeuroKit2-style points per lead out.

Every method (NeuroKit2, the TimesFM heads, the U-Net) returns, per lead, NeuroKit2's eleven keys
with one value per beat and NaN where that beat lacks the wave. Waves are grouped into beats the way
NeuroKit2 reports them: one beat per QRS, the nearest P before it and the nearest T after it
(decision 40). P or T waves with no QRS neighbour in their interval are orphans: dropped and counted.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import Wave

KEYS = (
    "ECG_P_Onsets",
    "ECG_P_Peaks",
    "ECG_P_Offsets",
    "ECG_R_Onsets",
    "ECG_Q_Peaks",
    "ECG_R_Peaks",
    "ECG_S_Peaks",
    "ECG_R_Offsets",
    "ECG_T_Onsets",
    "ECG_T_Peaks",
    "ECG_T_Offsets",
)
POINT_TYPES = ("P_on", "P_peak", "P_off", "QRS_on", "QRS_peak", "QRS_off", "T_on", "T_peak", "T_off")
KEY_FOR_POINT = {
    "P_on": "ECG_P_Onsets",
    "P_peak": "ECG_P_Peaks",
    "P_off": "ECG_P_Offsets",
    "QRS_on": "ECG_R_Onsets",
    "QRS_peak": "ECG_R_Peaks",
    "QRS_off": "ECG_R_Offsets",
    "T_on": "ECG_T_Onsets",
    "T_peak": "ECG_T_Peaks",
    "T_off": "ECG_T_Offsets",
}
CSE_POINTS = ("P_on", "P_off", "QRS_on", "QRS_off", "T_off")
CSE_2SD_MS = {"P_on": 10.2, "P_off": 12.7, "QRS_on": 6.5, "QRS_off": 11.6, "T_off": 30.6}


@dataclass(frozen=True)
class LeadPoints:
    """One lead's output: ``points[key]`` has one value per beat (sample index, NaN where absent)."""

    points: dict[str, list[float]]
    orphans: int = 0  # P or T waves dropped for lack of a QRS neighbour
    failed: bool = False  # the method raised on this lead; ``points`` are empty

    @property
    def n_beats(self) -> int:
        return len(self.points["ECG_R_Peaks"])

    def to_dict(self) -> dict[str, object]:
        """JSON-safe: NaN becomes null."""
        points = {key: [None if math.isnan(v) else v for v in values] for key, values in self.points.items()}
        return {"points": points, "orphans": self.orphans, "failed": self.failed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LeadPoints":
        points = {key: [math.nan if v is None else float(v) for v in d["points"][key]] for key in KEYS}
        return cls(points, orphans=int(d.get("orphans", 0)), failed=bool(d.get("failed", False)))


class Delineator(Protocol):
    name: str

    def delineate(self, signal: NDArray[np.float64], fs: int) -> dict[str, LeadPoints]:
        """``signal`` is (12, samples) in mV in LEADS order; returns lead name -> points."""
        ...


class Beat(NamedTuple):
    p: Wave | None
    qrs: Wave
    t: Wave | None


def empty_points(failed: bool = False) -> LeadPoints:
    return LeadPoints({key: [] for key in KEYS}, failed=failed)


def group_beats(waves: Sequence[Wave]) -> tuple[list[Beat], int]:
    """Beats in time order plus the number of orphan P/T waves (decision 40)."""
    qrs = sorted((w for w in waves if w.kind == "QRS"), key=lambda w: w.peak)
    others = [w for w in waves if w.kind != "QRS"]
    beats = []
    used = 0
    for k, q in enumerate(qrs):
        lo = qrs[k - 1].peak if k > 0 else -math.inf
        hi = qrs[k + 1].peak if k + 1 < len(qrs) else math.inf
        ps = [w for w in others if w.kind == "P" and lo < w.peak < q.peak]
        ts = [w for w in others if w.kind == "T" and q.peak < w.peak < hi]
        p = max(ps, key=lambda w: w.peak) if ps else None
        t = min(ts, key=lambda w: w.peak) if ts else None
        used += (p is not None) + (t is not None)
        beats.append(Beat(p, q, t))
    return beats, len(others) - used


def _value(sample: int | None) -> float:
    return math.nan if sample is None else float(sample)


def points_from_beats(beats: Sequence[Beat], q_peaks: Sequence[float] | None = None, s_peaks: Sequence[float] | None = None, orphans: int = 0) -> LeadPoints:
    n = len(beats)
    q = list(q_peaks) if q_peaks is not None else [math.nan] * n
    s = list(s_peaks) if s_peaks is not None else [math.nan] * n
    points: dict[str, list[float]] = {
        "ECG_P_Onsets": [_value(b.p.onset) if b.p else math.nan for b in beats],
        "ECG_P_Peaks": [_value(b.p.peak) if b.p else math.nan for b in beats],
        "ECG_P_Offsets": [_value(b.p.offset) if b.p else math.nan for b in beats],
        "ECG_R_Onsets": [_value(b.qrs.onset) for b in beats],
        "ECG_Q_Peaks": q,
        "ECG_R_Peaks": [_value(b.qrs.peak) for b in beats],
        "ECG_S_Peaks": s,
        "ECG_R_Offsets": [_value(b.qrs.offset) for b in beats],
        "ECG_T_Onsets": [_value(b.t.onset) if b.t else math.nan for b in beats],
        "ECG_T_Peaks": [_value(b.t.peak) if b.t else math.nan for b in beats],
        "ECG_T_Offsets": [_value(b.t.offset) if b.t else math.nan for b in beats],
    }
    return LeadPoints(points, orphans=orphans)


def points_from_waves(waves: Sequence[Wave]) -> LeadPoints:
    """NeuroKit2-style points from a list of waves (Q and S peaks left NaN)."""
    beats, orphans = group_beats(waves)
    return points_from_beats(beats, orphans=orphans)
