"""PTB-XL and PTB-XL+ for the pretraining experiment (decisions 72 to 77).

Three parts:
- the label check (decision 74): which ECGs' ECGDeli intervals agree well enough with GE 12SL and Uni-G;
- the stratified sample (decision 76): 4,500 + 500 ECGs in PTB-XL's own diagnosis and rhythm mix;
- ECGDeli's per-lead marks as step 1's waves (decision 75), so ``labels.labels_for`` paints and masks
  them exactly as it does LUDB's marks.
"""

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecgdelin.dataset import Wave

# PTB-XL+ names of the 12 leads, in step 1's LEADS order (i, ii, iii, avr, avl, avf, v1 ... v6).
PTB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INTERVALS = ("PR", "QRS", "QT")
REFERENCES = ("sl", "ug")  # GE 12SL, Uni-G
_GLOBAL = {"PR": "PR_Int_Global", "QRS": "QRS_Dur_Global", "QT": "QT_Int_Global"}
# ECGDeli per-lead interval columns; its PR_Int runs to the R peak, so PR uses PQ_Int (P onset to QRS onset).
_ECGDELI = {"PR": "PQ_Int_{}", "QRS": "QRS_Dur_{}", "QT": "QT_Int_{}"}
# Records with no ECGDeli fiducial files at all (PTB-XL+ 1.0.1 listing, 2026-10-15).
NO_ECGDELI_FILES = (7120, 13196, 18247, 21602)
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


# ---------------------------------------------------------------- the label check (decision 74)


def interval_table(db: pd.DataFrame, sl: pd.DataFrame, ug: pd.DataFrame, de: pd.DataFrame) -> pd.DataFrame:
    """One row per ECG (index ecg_id): strat_fold and PR, QRS, QT from 12SL (sl_*), Uni-G (ug_*) and
    ECGDeli (de_*, the median over the 12 leads). Uni-G's PR counts only where it found P waves."""
    ids = db.index
    sl, ug, de = (frame.reindex(ids) for frame in (sl, ug, de))
    table = pd.DataFrame(index=ids)
    table["strat_fold"] = db["strat_fold"]
    for k, column in _GLOBAL.items():
        table[f"sl_{k}"] = sl[column]
        table[f"ug_{k}"] = ug[column]
        table[f"de_{k}"] = de[[_ECGDELI[k].format(lead) for lead in PTB_LEADS]].median(axis=1)
    table.loc[ug["P_Found_Global"] == 0, "ug_PR"] = np.nan
    return table


@dataclass(frozen=True)
class CheckParams:
    """ECGDeli's median offset from each reference (``offsets["QRS/sl"]``, ms) and, per interval, the
    limit: the 99th percentile of |12SL - Uni-G| (ms). Fitted on ``folds`` only."""

    offsets: dict[str, float]
    limits: dict[str, float]
    folds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {"offsets_ms": self.offsets, "limits_ms": self.limits, "fitted_on_folds": list(self.folds)}


def fit_check(table: pd.DataFrame, folds: Sequence[int], quantile: float = 0.99) -> CheckParams:
    sub = table[table["strat_fold"].isin(folds)]
    offsets = {f"{k}/{ref}": float((sub[f"de_{k}"] - sub[f"{ref}_{k}"]).median()) for k in INTERVALS for ref in REFERENCES}
    limits = {k: float((sub[f"sl_{k}"] - sub[f"ug_{k}"]).abs().dropna().quantile(quantile)) for k in INTERVALS}
    return CheckParams(offsets, limits, tuple(int(f) for f in folds))


def failed_intervals(table: pd.DataFrame, params: CheckParams) -> pd.DataFrame:
    """Per ECG and interval: True where ECGDeli, its offset removed, differs from both 12SL and Uni-G by
    more than the limit. An interval with any of the three values missing cannot fail."""
    out = pd.DataFrame(index=table.index)
    for k in INTERVALS:
        gaps = [(table[f"de_{k}"] - params.offsets[f"{k}/{ref}"] - table[f"{ref}_{k}"]).abs() for ref in REFERENCES]
        checked = gaps[0].notna() & gaps[1].notna()
        out[k] = checked & (gaps[0] > params.limits[k]) & (gaps[1] > params.limits[k])
    return out


def passes_check(table: pd.DataFrame, params: CheckParams) -> pd.Series:
    return ~failed_intervals(table, params).any(axis=1)


# ---------------------------------------------------------------- the sample (decision 76)


def scp_codes(db: pd.DataFrame) -> pd.Series:
    return db["scp_codes"].apply(ast.literal_eval)


def superclasses(db: pd.DataFrame, scp: pd.DataFrame) -> pd.Series:
    """The diagnostic superclasses (NORM, MI, STTC, CD, HYP) of each ECG's diagnostic statements."""
    diagnostic = scp[scp["diagnostic"] == 1]["diagnostic_class"].to_dict()
    return scp_codes(db).apply(lambda codes: tuple(sorted({diagnostic[c] for c in codes if c in diagnostic})))


def af_or_flutter(db: pd.DataFrame) -> pd.Series:
    return scp_codes(db).apply(lambda codes: "AFIB" in codes or "AFLT" in codes)


def strata(db: pd.DataFrame, scp: pd.DataFrame) -> pd.Series:
    """One stratum per ECG: its rhythm group and its set of superclasses, e.g. ``AF|CD+MI`` or ``other|NORM``."""
    sup = superclasses(db, scp)
    af = af_or_flutter(db)
    return pd.Series([f"{'AF' if a else 'other'}|{'+'.join(s) or 'none'}" for a, s in zip(af, sup)], index=db.index)


def allocate(reference: pd.Series, available: pd.Series, n: int) -> dict[str, int]:
    """Split ``n`` across strata in proportion to ``reference`` counts (largest remainder), never more
    than ``available`` in a stratum; a shortfall is shared out the same way among strata with room."""
    available = available.reindex(reference.index, fill_value=0)
    take = {s: 0 for s in reference.index}
    remaining = n
    open_strata = [s for s in reference.index if available[s] > 0]
    while remaining > 0 and open_strata:
        weights = reference[open_strata].astype(float)
        exact = weights / weights.sum() * remaining
        base = {s: int(math.floor(exact[s])) for s in open_strata}
        leftover = remaining - sum(base.values())
        for s in sorted(open_strata, key=lambda s: (-(exact[s] - base[s]), s))[:leftover]:
            base[s] += 1
        for s in open_strata:
            add = min(base[s], int(available[s]) - take[s])
            take[s] += add
            remaining -= add
        open_strata = [s for s in open_strata if take[s] < available[s]]
    return {s: k for s, k in take.items() if k > 0}


def stratified_sample(pool: pd.Series, reference: pd.Series, n: int, seed: int) -> list[int]:
    """``pool``: stratum per eligible ECG (index ecg_id); ``reference``: stratum per ECG of the mix to
    match. Returns ``n`` ecg_ids, sorted, drawn without replacement inside each stratum."""
    counts = allocate(reference.value_counts(), pool.value_counts(), n)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for stratum in sorted(counts):
        ids = np.sort(pool.index[pool == stratum].to_numpy())
        chosen.extend(int(i) for i in rng.choice(ids, size=counts[stratum], replace=False))
    return sorted(chosen)


# ---------------------------------------------------------------- ECGDeli marks as waves (decision 75)

POINT_NAMES = {
    "p-wave onset": "P_on",
    "p-wave peak": "P_peak",
    "p-wave offset": "P_off",
    "QRS onset": "QRS_on",
    "R peak": "R",
    "QRS offset": "QRS_off",
    "t-wave onset": "T_on",
    "t-wave peak": "T_peak",
    "t-wave offset": "T_off",
}
_WAVE_POINTS = {"P": ("P_on", "P_peak", "P_off"), "QRS": ("QRS_on", "R", "QRS_off"), "T": ("T_on", "T_peak", "T_off")}


def read_ecgdeli_lead(stem: Path) -> dict[str, list[int]]:
    """One PTB-XL+ lead file (``<dir>/<id>_points_lead_<LEAD>`` without ``.atr``): point -> samples.
    Returns {} when the file does not exist."""
    import wfdb

    if not Path(f"{stem}.atr").exists():
        return {}
    annotation = wfdb.rdann(str(stem), "atr")
    points: dict[str, list[int]] = {}
    for name, sample in zip(annotation.aux_note, annotation.sample):
        key = POINT_NAMES.get(name.strip())
        if key is not None:
            points.setdefault(key, []).append(int(sample))
    return {key: sorted(values) for key, values in points.items()}


def _usable(value: float | None, n_samples: int) -> float | None:
    """ECGDeli writes placeholders where it has no mark: sample 1 (or 0) at the start, 4998 to 5000 at
    the end of a 5000-sample record. Those count as missing."""
    if value is None or not math.isfinite(value) or value < 2 or value > n_samples - 3:
        return None
    return value


@dataclass(frozen=True)
class LeadWaves:
    waves: list[Wave]
    n_beats: int
    n_masked_beats: int  # beats turned into incomplete waves because their marks overlap or are out of order
    aligned: bool  # False when the point types have unequal counts; then the whole lead is left unlabelled


def ecgdeli_waves(points: Mapping[str, Sequence[int]], n_samples: int, af: bool) -> LeadWaves:
    """ECGDeli's marks of one lead as step 1's waves.

    ECGDeli writes the 12 points of every beat, so the k-th mark of each type belongs to beat k. A beat
    whose marks overlap or are out of order (inside a wave, between its waves, or its T ending after
    the next beat starts) becomes three incomplete waves, so ``labels_for`` masks it; both beats of a
    T-into-next-beat overlap are masked. Placeholder marks are missing. In AF and flutter ECGs the P
    marks are dropped, so the P region is background, as in LUDB.
    """
    if not points:
        return LeadWaves([], 0, 0, True)
    n = len(points.get("R", []))
    needed = [p for wave in _WAVE_POINTS.values() for p in wave]
    if n == 0 or any(len(points.get(p, [])) != n for p in needed):
        return LeadWaves([], n, 0, False)
    beats = [{p: _usable(float(points[p][k]), n_samples) for p in needed} for k in range(n)]
    if af:
        for beat in beats:
            beat.update(P_on=None, P_peak=None, P_off=None)

    def before(a: float | None, b: float | None) -> bool:
        return a is None or b is None or a < b

    bad = [False] * n
    for k, beat in enumerate(beats):
        for on, peak, off in _WAVE_POINTS.values():
            if not before(beat[on], beat[off]):
                bad[k] = True
            if beat[peak] is not None and beat[on] is not None and beat[off] is not None and not beat[on] <= beat[peak] <= beat[off]:
                bad[k] = True
        if not (before(beat["P_off"], beat["QRS_on"]) and before(beat["QRS_off"], beat["T_on"]) and before(beat["P_on"], beat["QRS_on"])):
            bad[k] = True
        if k + 1 < n:
            nxt = beats[k + 1]
            starts = [v for v in (nxt["P_on"], nxt["QRS_on"]) if v is not None]
            ends = [v for v in (beat["T_off"], beat["QRS_off"]) if v is not None]
            if starts and ends and not max(ends) < min(starts):
                bad[k] = bad[k + 1] = True

    waves: list[Wave] = []
    for k, beat in enumerate(beats):
        for kind, (on, peak_key, off) in _WAVE_POINTS.items():
            onset, peak, offset = beat[on], beat[peak_key], beat[off]
            if onset is None and peak is None and offset is None:
                continue
            if peak is None or (onset is not None and offset is not None and not onset <= peak <= offset):
                known = [v for v in (onset, offset) if v is not None]
                peak = sum(known) / len(known) if known else peak
            if peak is None:
                continue
            if bad[k]:
                onset = offset = None
            waves.append(Wave(kind, None if onset is None else int(onset), int(round(peak)), None if offset is None else int(offset)))
    waves.sort(key=lambda w: w.peak)
    return LeadWaves(waves, n, sum(bad), True)
