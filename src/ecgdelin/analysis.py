"""Development analysis for the M3 deep dive (decision 60).

Per-lead errors live in the scorer (``score.errors_by_lead``). This module adds what the report needs
on top: why a class map's orphan waves were dropped, the cardiologists' marks pushed through the
decoder, and one person's primary SD.
"""

import math

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import LEADS, Recording
from ecgdelin.decode import regions_to_points, waves_from_classes
from ecgdelin.delineate import CSE_POINTS, LeadPoints, group_beats
from ecgdelin.labels import labels_for
from ecgdelin.score import PersonScore

ORPHAN_KINDS = ("edge", "extra", "unanchored")


def orphan_breakdown(classes: NDArray[np.int8], rec: Recording, signal: NDArray[np.floating]) -> dict[str, int]:
    """Why each orphan P or T wave of a (12, samples) class map was dropped (decision 40).

    ``edge``: its peak lies before the lead's first mark or after its last, where no reference exists.
    ``unanchored``: inside the marked span but with no predicted QRS on the side it needs (a P after
    the last QRS, a T before the first), which points at a missed QRS.
    ``extra``: inside the span with a QRS neighbour whose slot a nearer wave took: a fragment or a
    doubled wave.
    """
    out = dict.fromkeys(ORPHAN_KINDS, 0)
    for k, lead in enumerate(LEADS):
        waves = waves_from_classes(classes[k], np.asarray(signal[k], dtype=np.float64), rec.fs)
        beats, _ = group_beats(waves)
        attached = {b.p for b in beats if b.p} | {b.t for b in beats if b.t}
        marks = [s for w in rec.marks[lead] for s in (w.onset, w.peak, w.offset) if s is not None]
        first, last = (min(marks), max(marks)) if marks else (math.inf, -math.inf)
        qrs_peaks = [w.peak for w in waves if w.kind == "QRS"]
        for w in waves:
            if w.kind == "QRS" or w in attached:
                continue
            if w.peak < first or w.peak > last:
                out["edge"] += 1
            elif (w.kind == "P" and not any(q > w.peak for q in qrs_peaks)) or (w.kind == "T" and not any(q < w.peak for q in qrs_peaks)):
                out["unanchored"] += 1
            else:
                out["extra"] += 1
    return out


def roundtrip_points(rec: Recording, signal: NDArray[np.floating]) -> dict[str, LeadPoints]:
    """The cardiologists' marks pushed through the labels and the decoder on ``signal`` (12, samples),
    the prepared signal the trained models saw. Shows what the decoder alone does to the points."""
    n = signal.shape[1]
    return {
        lead: regions_to_points(labels_for(rec.marks[lead], n).classes, np.asarray(signal[k], dtype=np.float64), rec.fs)
        for k, lead in enumerate(LEADS)
    }


def person_primary_sd(score: PersonScore) -> float:
    """Mean over the CSE points of the person's pooled error SD; points with fewer than two matched
    pairs are left out, and NaN means none had two."""
    sds = [float(np.std(score.errors_ms[p])) for p in CSE_POINTS if score.errors_ms[p].size >= 2]
    return float(np.mean(sds)) if sds else math.nan
