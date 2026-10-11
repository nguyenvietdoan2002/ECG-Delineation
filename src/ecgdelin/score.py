"""The scorer: predicted points against LUDB marks for a set of recordings.

Per lead and point type, predictions are matched one-to-one to the marks within a tolerance window,
inside the marked span only (decisions 41 and 44). Error = prediction - reference in ms. SDs are
population SDs (decision 45). Confidence intervals and the stop rule come from a bootstrap over
people (decision 28).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import LEADS, Recording, Wave
from ecgdelin.delineate import CSE_2SD_MS, CSE_POINTS, KEY_FOR_POINT, POINT_TYPES, LeadPoints
from ecgdelin.labels import CLASSES, labels_for
from ecgdelin.splits import rhythm_group

TOLERANCES_MS = (150, 40)
N_DRAWS = 10_000
WAVE_CLASSES = ("P", "QRS", "T")
REGION_KEYS = (("P", "ECG_P_Onsets", "ECG_P_Offsets"), ("T", "ECG_T_Onsets", "ECG_T_Offsets"), ("QRS", "ECG_R_Onsets", "ECG_R_Offsets"))


@dataclass(frozen=True)
class ToleranceStats:
    tp: int
    fn: int
    fp: int

    @property
    def se(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def ppv(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fn + self.fp)

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fn": self.fn, "fp": self.fp, "se": _num(self.se), "ppv": _num(self.ppv), "f1": _num(self.f1)}


@dataclass(frozen=True)
class PointStats:
    at: dict[int, ToleranceStats]  # tolerance in ms -> detection counts
    mean_ms: float  # pooled over all matched pairs at the loosest tolerance
    sd_ms: float  # pooled population SD of those errors
    sd_within_ms: float  # mean over recording-leads (with >= 2 pairs) of the per-lead SD
    cse_2sd_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": {str(t): s.to_dict() for t, s in self.at.items()},
            "mean_ms": _num(self.mean_ms),
            "sd_ms": _num(self.sd_ms),
            "sd_within_ms": _num(self.sd_within_ms),
            "cse_2sd_ms": self.cse_2sd_ms,
        }


@dataclass(frozen=True)
class Report:
    n_people: int
    points: dict[str, PointStats]
    primary_sd_ms: float  # mean over the five CSE points of the pooled error SD
    primary_ci95: tuple[float, float]  # bootstrap over people
    miou: dict[str, float]  # "P", "QRS", "T" and "mean", over masked samples
    failed_leads: int
    orphans: int
    by_rhythm: dict[str, "Report"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_people": self.n_people,
            "primary_sd_ms": _num(self.primary_sd_ms),
            "primary_ci95": [_num(x) for x in self.primary_ci95],
            "points": {p: s.to_dict() for p, s in self.points.items()},
            "miou": {k: _num(v) for k, v in self.miou.items()},
            "failed_leads": self.failed_leads,
            "orphans": self.orphans,
            "by_rhythm": {g: r.to_dict() for g, r in self.by_rhythm.items()},
        }


@dataclass(frozen=True)
class Verdict:
    """The stop rule for method a against method b (decision 28)."""

    diff_ci95: tuple[float, float]  # 95% interval of primary(a) - primary(b); lower SD is better
    a_better: bool  # the interval lies entirely below zero
    f1_diff_ci95: tuple[float, float]  # 95% interval of F1(a) - F1(b) at the loosest tolerance, CSE points
    f1_safeguard_ok: bool  # that interval does not lie entirely below zero
    a_wins: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "diff_ci95": [_num(x) for x in self.diff_ci95],
            "a_better": self.a_better,
            "f1_diff_ci95": [_num(x) for x in self.f1_diff_ci95],
            "f1_safeguard_ok": self.f1_safeguard_ok,
            "a_wins": self.a_wins,
        }


@dataclass(frozen=True)
class PersonScore:
    """One recording's contribution, kept per person so that bootstraps can resample people."""

    rec_id: str
    rhythm_group: str
    errors_ms: dict[str, NDArray[np.float64]]  # point type -> errors of matched pairs (loosest tolerance)
    counts: dict[int, dict[str, ToleranceStats]]  # tolerance -> point type -> counts
    within_sd: dict[str, list[float]]  # point type -> per-lead SD for leads with >= 2 pairs
    overlap: dict[str, tuple[int, int]]  # wave class -> (intersection, union) in masked samples
    failed_leads: int
    orphans: int


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def _num(x: float | None) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


def match(ref: NDArray[np.int64], pred: NDArray[np.int64], tol: int) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, nearest pair first, |pred - ref| <= tol (decision 41)."""
    if ref.size == 0 or pred.size == 0:
        return []
    d = pred[None, :] - ref[:, None]
    i, j = np.nonzero(np.abs(d) <= tol)
    order = np.lexsort((j, i, np.abs(d[i, j])))
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    pairs = []
    for k in order:
        a, b = int(i[k]), int(j[k])
        if a in used_ref or b in used_pred:
            continue
        used_ref.add(a)
        used_pred.add(b)
        pairs.append((a, b))
    return pairs


def _reference_points(waves: Sequence[Wave], mask: NDArray[np.bool_]) -> dict[str, NDArray[np.int64]]:
    points: dict[str, list[int]] = {p: [] for p in POINT_TYPES}
    for w in waves:
        if w.onset is None or w.offset is None:
            continue
        for part, sample in (("on", w.onset), ("peak", w.peak), ("off", w.offset)):
            if mask[sample]:
                points[f"{w.kind}_{part}"].append(sample)
    return {p: np.array(sorted(v), dtype=np.int64) for p, v in points.items()}


def _predicted_points(lead: LeadPoints, n_samples: int) -> dict[str, NDArray[np.int64]]:
    out = {}
    for point in POINT_TYPES:
        samples = []
        for value in lead.points[KEY_FOR_POINT[point]]:
            if math.isfinite(value) and 0 <= int(round(value)) < n_samples:
                samples.append(int(round(value)))
        out[point] = np.array(sorted(samples), dtype=np.int64)
    return out


def _predicted_classes(lead: LeadPoints, n_samples: int) -> NDArray[np.int8]:
    """Regions rebuilt from each beat's onset/offset pairs; QRS painted last so it wins overlaps."""
    classes = np.zeros(n_samples, dtype=np.int8)
    for kind, on_key, off_key in REGION_KEYS:
        for on, off in zip(lead.points[on_key], lead.points[off_key]):
            if math.isfinite(on) and math.isfinite(off):
                a, b = max(0, int(round(on))), min(n_samples - 1, int(round(off)))
                if a <= b:
                    classes[a : b + 1] = CLASSES.index(kind)
    return classes


def _match_lead(
    ref: Mapping[str, NDArray[np.int64]], got: Mapping[str, NDArray[np.int64]], tol: int, mask: NDArray[np.bool_], fs: int
) -> tuple[dict[str, list[float]], dict[str, ToleranceStats]]:
    """One lead at one tolerance: matched errors in ms and counts per point type."""
    errors: dict[str, list[float]] = {}
    counts: dict[str, ToleranceStats] = {}
    for point in POINT_TYPES:
        pairs = match(ref[point], got[point], tol)
        matched_pred = {b for _, b in pairs}
        # An unmatched prediction counts as a false positive only inside the marked span;
        # a prediction just outside the span can still match a mark at the span's edge.
        fp = sum(1 for b, s in enumerate(got[point]) if b not in matched_pred and mask[s])
        counts[point] = ToleranceStats(len(pairs), ref[point].size - len(pairs), fp)
        errors[point] = [(got[point][b] - ref[point][a]) * 1000 / fs for a, b in pairs]
    return errors, counts


@dataclass(frozen=True)
class LeadScore:
    """One lead's matched errors and counts at one tolerance."""

    errors_ms: dict[str, NDArray[np.float64]]
    counts: dict[str, ToleranceStats]


def errors_by_lead(rec: Recording, pred: Mapping[str, LeadPoints], tol_ms: int = max(TOLERANCES_MS)) -> dict[str, LeadScore]:
    """Per-lead errors and counts for one recording, with the same matching as ``score_recording``."""
    n = rec.signal.shape[1]
    tol = int(round(tol_ms * rec.fs / 1000))
    empty = {p: np.array([], dtype=np.int64) for p in POINT_TYPES}
    out = {}
    for lead in LEADS:
        labels = labels_for(rec.marks[lead], n)
        ref = _reference_points(rec.marks[lead], labels.mask)
        got = _predicted_points(pred[lead], n) if lead in pred else empty
        errors, counts = _match_lead(ref, got, tol, labels.mask, rec.fs)
        out[lead] = LeadScore({p: np.array(v, dtype=np.float64) for p, v in errors.items()}, counts)
    return out


def score_recording(rec: Recording, pred: Mapping[str, LeadPoints], tolerances_ms: Sequence[int] = TOLERANCES_MS) -> PersonScore:
    tols = {t: int(round(t * rec.fs / 1000)) for t in tolerances_ms}
    loosest = max(tolerances_ms)
    errors: dict[str, list[float]] = {p: [] for p in POINT_TYPES}
    within: dict[str, list[float]] = {p: [] for p in POINT_TYPES}
    counts = {t: {p: [0, 0, 0] for p in POINT_TYPES} for t in tolerances_ms}
    overlap = {c: [0, 0] for c in WAVE_CLASSES}
    n = rec.signal.shape[1]
    empty = {p: np.array([], dtype=np.int64) for p in POINT_TYPES}
    for lead in LEADS:
        labels = labels_for(rec.marks[lead], n)
        mask = labels.mask
        ref = _reference_points(rec.marks[lead], mask)
        got = _predicted_points(pred[lead], n) if lead in pred else empty
        for t in tolerances_ms:
            lead_errors, lead_counts = _match_lead(ref, got, tols[t], mask, rec.fs)
            for point in POINT_TYPES:
                c = counts[t][point]
                c[0] += lead_counts[point].tp
                c[1] += lead_counts[point].fn
                c[2] += lead_counts[point].fp
                if t == loosest:
                    errors[point] += lead_errors[point]
                    if len(lead_errors[point]) >= 2:
                        within[point].append(float(np.std(lead_errors[point])))
        if lead in pred:
            predicted = _predicted_classes(pred[lead], n)
            for kind in WAVE_CLASSES:
                k = CLASSES.index(kind)
                overlap[kind][0] += int(np.sum((predicted == k) & (labels.classes == k) & mask))
                overlap[kind][1] += int(np.sum(((predicted == k) | (labels.classes == k)) & mask))
        else:
            for kind in WAVE_CLASSES:
                overlap[kind][1] += int(np.sum((labels.classes == CLASSES.index(kind)) & mask))
    return PersonScore(
        rec.id,
        rhythm_group(rec.rhythm),
        {p: np.array(v, dtype=np.float64) for p, v in errors.items()},
        {t: {p: ToleranceStats(*c) for p, c in per_point.items()} for t, per_point in counts.items()},
        within,
        {c: (v[0], v[1]) for c, v in overlap.items()},
        sum(1 for lead in LEADS if lead in pred and pred[lead].failed),
        sum(pred[lead].orphans for lead in LEADS if lead in pred),
    )


def _draw_counts(n_people: int, n_draws: int, seed: int) -> NDArray[np.float64]:
    """How often each person appears in each bootstrap draw of ``n_people`` with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_people, size=(n_draws, n_people))
    counts = np.zeros((n_draws, n_people))
    np.add.at(counts, (np.arange(n_draws)[:, None], idx), 1.0)
    return counts


def _primary_per_draw(scores: Sequence[PersonScore], counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """The primary metric of every draw, from per-person sums (population SD decomposes exactly)."""
    sds = []
    for point in CSE_POINTS:
        n = counts @ np.array([s.errors_ms[point].size for s in scores], dtype=np.float64)
        s1 = counts @ np.array([s.errors_ms[point].sum() for s in scores])
        s2 = counts @ np.array([np.square(s.errors_ms[point]).sum() for s in scores])
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 / n - (s1 / n) ** 2
        sds.append(np.sqrt(np.maximum(var, 0.0)))
    return np.asarray(np.mean(sds, axis=0))


def _f1_per_draw(scores: Sequence[PersonScore], counts: NDArray[np.float64], tol: int) -> NDArray[np.float64]:
    f1s = []
    for point in CSE_POINTS:
        tp = counts @ np.array([s.counts[tol][point].tp for s in scores], dtype=np.float64)
        fn = counts @ np.array([s.counts[tol][point].fn for s in scores], dtype=np.float64)
        fp = counts @ np.array([s.counts[tol][point].fp for s in scores], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            f1s.append(2 * tp / (2 * tp + fn + fp))
    return np.asarray(np.mean(f1s, axis=0))


def _ci95(values: NDArray[np.float64]) -> tuple[float, float]:
    if values.size == 0 or np.all(np.isnan(values)):
        return (math.nan, math.nan)
    return (float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5)))


def summarize(
    scores: Sequence[PersonScore],
    tolerances_ms: Sequence[int] = TOLERANCES_MS,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> Report:
    points = {}
    for point in POINT_TYPES:
        errors = np.concatenate([s.errors_ms[point] for s in scores]) if scores else np.array([])
        within = [v for s in scores for v in s.within_sd[point]]
        at = {}
        for t in tolerances_ms:
            tp = sum(s.counts[t][point].tp for s in scores)
            fn = sum(s.counts[t][point].fn for s in scores)
            fp = sum(s.counts[t][point].fp for s in scores)
            at[t] = ToleranceStats(tp, fn, fp)
        points[point] = PointStats(
            at,
            mean_ms=float(errors.mean()) if errors.size else math.nan,
            sd_ms=float(errors.std()) if errors.size else math.nan,
            sd_within_ms=float(np.mean(within)) if within else math.nan,
            cse_2sd_ms=CSE_2SD_MS.get(point),
        )
    # All five points exist for any whole test population; a subgroup without P waves (AF/flutter)
    # is summarised over the points it has, so by-rhythm tables show a number rather than nothing.
    sds = [points[p].sd_ms for p in CSE_POINTS]
    primary = float(np.nanmean(sds)) if any(not math.isnan(s) for s in sds) else math.nan
    ci = _ci95(_primary_per_draw(scores, _draw_counts(len(scores), n_draws, seed))) if scores else (math.nan, math.nan)
    miou = {
        kind: _ratio(sum(s.overlap[kind][0] for s in scores), sum(s.overlap[kind][1] for s in scores))
        for kind in WAVE_CLASSES
    }
    miou["mean"] = float(np.nanmean([miou[k] for k in WAVE_CLASSES])) if not all(math.isnan(miou[k]) for k in WAVE_CLASSES) else math.nan
    return Report(
        len(scores),
        points,
        primary,
        ci,
        miou,
        sum(s.failed_leads for s in scores),
        sum(s.orphans for s in scores),
    )


def score(
    recordings: Sequence[Recording],
    predictions: Mapping[str, Mapping[str, LeadPoints]],
    tolerances_ms: Sequence[int] = TOLERANCES_MS,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> Report:
    """Score one method: ``predictions[recording id][lead]``. Includes a per-rhythm breakdown."""
    scores = [score_recording(rec, predictions.get(rec.id, {}), tolerances_ms) for rec in recordings]
    report = summarize(scores, tolerances_ms, n_draws, seed)
    by_rhythm = {}
    for group in sorted({s.rhythm_group for s in scores}):
        by_rhythm[group] = summarize([s for s in scores if s.rhythm_group == group], tolerances_ms, n_draws, seed)
    return Report(
        report.n_people,
        report.points,
        report.primary_sd_ms,
        report.primary_ci95,
        report.miou,
        report.failed_leads,
        report.orphans,
        by_rhythm,
    )


Predictions = Mapping[str, Mapping[str, LeadPoints]]  # recording id -> lead -> points


def _per_seed(recordings: Sequence[Recording], seeds: Sequence[Predictions], tolerances_ms: Sequence[int]) -> list[list[PersonScore]]:
    return [[score_recording(rec, preds.get(rec.id, {}), tolerances_ms) for rec in recordings] for preds in seeds]


def seed_averaged_primary(
    recordings: Sequence[Recording],
    predictions_seeds: Sequence[Predictions],
    tolerances_ms: Sequence[int] = TOLERANCES_MS,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """The primary metric averaged over a method's seeds, with its 95% bootstrap interval over people
    (the metric is computed per seed inside every draw, then averaged; decision 64)."""
    scores = _per_seed(recordings, predictions_seeds, tolerances_ms)
    counts = _draw_counts(len(recordings), n_draws, seed)
    per_draw = np.mean([_primary_per_draw(s, counts) for s in scores], axis=0)
    point = float(np.mean([summarize(s, tolerances_ms, n_draws=0).primary_sd_ms for s in scores]))
    return point, _ci95(np.asarray(per_draw))


def compare_seeds(
    recordings: Sequence[Recording],
    predictions_a: Sequence[Predictions],
    predictions_b: Sequence[Predictions],
    tolerances_ms: Sequence[int] = TOLERANCES_MS,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> Verdict:
    """Paired bootstrap over the same people, several seeds per method: in every draw each method's
    primary metric (and F1 at the loosest tolerance) is computed per seed and averaged over its seeds
    before the difference is taken (decisions 28 and 64). Does method a beat method b?"""
    scores_a = _per_seed(recordings, predictions_a, tolerances_ms)
    scores_b = _per_seed(recordings, predictions_b, tolerances_ms)
    counts = _draw_counts(len(recordings), n_draws, seed)
    loosest = max(tolerances_ms)
    primary_a = np.mean([_primary_per_draw(s, counts) for s in scores_a], axis=0)
    primary_b = np.mean([_primary_per_draw(s, counts) for s in scores_b], axis=0)
    f1_a = np.mean([_f1_per_draw(s, counts, loosest) for s in scores_a], axis=0)
    f1_b = np.mean([_f1_per_draw(s, counts, loosest) for s in scores_b], axis=0)
    diff = _ci95(np.asarray(primary_a - primary_b))
    f1_diff = _ci95(np.asarray(f1_a - f1_b))
    a_better = diff[1] < 0
    f1_ok = not f1_diff[1] < 0
    return Verdict(diff, a_better, f1_diff, f1_ok, a_better and f1_ok)


def compare(
    recordings: Sequence[Recording],
    predictions_a: Predictions,
    predictions_b: Predictions,
    tolerances_ms: Sequence[int] = TOLERANCES_MS,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> Verdict:
    """Paired bootstrap over the same people, one prediction set per method: does a beat b (decision 28)?"""
    return compare_seeds(recordings, [predictions_a], [predictions_b], tolerances_ms, n_draws, seed)
