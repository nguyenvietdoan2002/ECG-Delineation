"""RDB, the Resting ECG Segmentation Dataset (2,399 Chapman-Shaoxing ECGs, figshare 10.6084/m9.figshare.28892186,
CC BY 4.0), as pretraining data (decisions 84 and 90).

Each record is a 10 s, 500 Hz, 12-lead CSV in the column order I, II, III, aVR, aVL, aVF, V1 to V6
(the README says millivolts; the values are microvolts, R peaks reach 2,800) and one set of marks for
all 12 leads in ``<id>.all.txt``: rows ``TYPE,START,END`` with TYPE 0 = P, 1 = QRS, 2 = T and 0-based
inclusive sample indices, some fractional, some at or beyond the record's edges. The 2026-10-13 check
found these shared marks to be the 12-lead envelope (earliest onset, latest offset); the per-lead files
are small edits of them and are not used. Marks become step 1's waves under decision 75's rules: a wave
that overlaps its neighbour is masked with it, a mark at the edge or beyond it counts as missing, and
the peak is the span's midpoint. AF and flutter records carry no P marks, so their P region is
background, as in LUDB. The rhythm class is the letters of the record id: AF (flutter), AFIB, SB, SR,
SI, ST, AT, VT.
"""

import io
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import LEADS, Wave
from ecgdelin.ptbxl import LeadWaves, _usable

FS = 500
N_SAMPLES = 5000
COLUMNS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
KINDS = {0: "P", 1: "QRS", 2: "T"}
AF_CLASSES = ("AF", "AFIB")  # flutter and fibrillation; neither carries P marks


def rhythm_class(record_id: str) -> str:
    return "".join(c for c in record_id if c.isalpha())


def parse_marks(text: str) -> list[tuple[str, float, float]]:
    """``TYPE,START,END`` rows (header optional) -> (kind, start, end) sorted by start."""
    out = []
    for line in text.lstrip("﻿").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 3 or not cells[0] or cells[0].upper() == "TYPE":
            continue
        out.append((KINDS[int(float(cells[0]))], float(cells[1]), float(cells[2])))
    return sorted(out, key=lambda m: (m[1], m[2]))


def read_signal(text: str) -> NDArray[np.float32]:
    """The CSV's (samples, 12) microvolts -> (12, samples) float32 millivolts in LEADS order."""
    data = np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError(f"expected 12 columns, got {data.shape}")
    order = [COLUMNS.index(lead) for lead in LEADS]
    return (data[:, order].T / 1000.0).astype(np.float32)


def rdb_waves(marks: Sequence[tuple[str, float, float]], n_samples: int = N_SAMPLES) -> LeadWaves:
    """The record's marks as step 1's waves. A mark below sample 2 or above ``n_samples - 3`` counts as
    missing (decision 75's edge rule); a span that overlaps the next one, or runs backwards, is masked
    together with its neighbour; the peak is the rounded midpoint, kept inside the record."""
    rows = [(kind, float(start), float(end)) for kind, start, end in marks]
    bad = [end < start for _, start, end in rows]
    for k in range(len(rows) - 1):
        if rows[k + 1][1] <= rows[k][2]:
            bad[k] = bad[k + 1] = True
    waves = []
    for k, (kind, start, end) in enumerate(rows):
        onset = None if bad[k] else _usable(round(start), n_samples)
        offset = None if bad[k] else _usable(round(end), n_samples)
        peak = int(min(max(round((start + end) / 2), 0), n_samples - 1))
        if onset is not None and offset is not None and not onset <= peak <= offset:
            peak = (onset + offset) // 2
        waves.append(Wave(kind, None if onset is None else int(onset), peak, None if offset is None else int(offset)))
    waves.sort(key=lambda w: w.peak)
    return LeadWaves(waves, sum(kind == "QRS" for kind, _, _ in rows), sum(bad), True)


def stop_split(ids: Sequence[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out ``fraction`` of the records for stopping, the same share of every rhythm class (seeded)."""
    rng = np.random.default_rng(seed)
    stop: list[str] = []
    classes = sorted({rhythm_class(i) for i in ids})
    for cls in classes:
        members = sorted(i for i in ids if rhythm_class(i) == cls)
        n_stop = int(round(fraction * len(members)))
        stop.extend(members[k] for k in sorted(rng.choice(len(members), n_stop, replace=False)))
    held = set(stop)
    return [i for i in ids if i not in held], [i for i in ids if i in held]
