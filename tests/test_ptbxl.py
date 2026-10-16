import numpy as np
import pandas as pd

from ecgdelin.labels import CLASSES, labels_for
from ecgdelin.ptbxl import (
    PTB_LEADS,
    allocate,
    ecgdeli_waves,
    fit_check,
    interval_table,
    passes_check,
    stratified_sample,
)

# ---------------------------------------------------------------- the label check


def _frames(rows: dict[int, dict[str, float]]) -> tuple[pd.DataFrame, ...]:
    """rows: ecg_id -> fold, sl_/ug_/de_ values of PR, QRS, QT, and optionally ug_pfound."""
    ids = list(rows)
    db = pd.DataFrame({"strat_fold": [rows[i]["fold"] for i in ids]}, index=pd.Index(ids, name="ecg_id"))
    sl = pd.DataFrame({"PR_Int_Global": [rows[i]["sl_PR"] for i in ids], "QRS_Dur_Global": [rows[i]["sl_QRS"] for i in ids],
                       "QT_Int_Global": [rows[i]["sl_QT"] for i in ids]}, index=db.index)
    ug = pd.DataFrame({"PR_Int_Global": [rows[i]["ug_PR"] for i in ids], "QRS_Dur_Global": [rows[i]["ug_QRS"] for i in ids],
                       "QT_Int_Global": [rows[i]["ug_QT"] for i in ids], "P_Found_Global": [rows[i].get("ug_pfound", 1) for i in ids]},
                      index=db.index)
    de = pd.DataFrame(index=db.index)
    for lead in PTB_LEADS:
        de[f"PQ_Int_{lead}"] = [rows[i]["de_PR"] for i in ids]
        de[f"QRS_Dur_{lead}"] = [rows[i]["de_QRS"] for i in ids]
        de[f"QT_Int_{lead}"] = [rows[i]["de_QT"] for i in ids]
    return db, sl, ug, de


def _row(fold: int, de_qrs: float = 118.0, de_qt: float = 411.0, **over: float) -> dict[str, float]:
    row = {"fold": fold, "sl_PR": 160.0, "ug_PR": 162.0, "de_PR": 162.0, "sl_QRS": 90.0, "ug_QRS": 92.0, "de_QRS": de_qrs,
           "sl_QT": 400.0, "ug_QT": 404.0, "de_QT": de_qt}
    row.update(over)
    return row


def test_check_removes_the_offset_and_drops_only_ecgs_off_from_both() -> None:
    rows = {i: _row(1 if i < 40 else 9) for i in range(1, 60)}
    rows[60] = _row(1, de_qrs=180.0)  # 62 ms past the offset-corrected value: fails QRS
    rows[61] = _row(1, de_qt=600.0)  # fails QT
    rows[62] = _row(1, de_qt=600.0, sl_QT=np.nan)  # 12SL missing: QT cannot fail
    rows[63] = _row(10, de_qrs=900.0, sl_QRS=900.0, ug_QRS=300.0)  # fold 10 must not move the fit
    table = interval_table(*_frames(rows))
    params = fit_check(table, folds=range(1, 10))
    assert params.offsets["QRS/sl"] == 28.0 and params.offsets["QRS/ug"] == 26.0
    assert params.limits["QRS"] == 2.0  # |90 - 92| everywhere in folds 1-9
    keep = passes_check(table, params)
    assert keep[1] and keep[62]
    assert not keep[60] and not keep[61]


def test_unig_pr_without_p_waves_is_missing() -> None:
    rows = {i: _row(1) for i in range(1, 30)}
    rows[30] = _row(1, de_PR=300.0, ug_pfound=0)
    table = interval_table(*_frames(rows))
    assert np.isnan(table.loc[30, "ug_PR"])
    assert passes_check(table, fit_check(table, folds=[1]))[30]


# ---------------------------------------------------------------- the sample


def test_allocation_is_proportional_and_shares_out_shortfalls() -> None:
    reference = pd.Series({"A": 60, "B": 30, "C": 10})
    available = pd.Series({"A": 100, "B": 10, "C": 50})
    assert allocate(reference, available, 50) == {"A": 34, "B": 10, "C": 6}
    assert sum(allocate(reference, pd.Series({"A": 5, "B": 5, "C": 5}), 50).values()) == 15


def test_stratified_sample_is_seeded_and_stays_in_the_pool() -> None:
    pool = pd.Series(["A"] * 50 + ["B"] * 50, index=range(100, 200))
    reference = pd.Series(["A"] * 80 + ["B"] * 20)
    first = stratified_sample(pool, reference, 20, seed=0)
    assert first == stratified_sample(pool, reference, 20, seed=0)
    assert len(first) == 20 and set(first) <= set(pool.index)
    assert sum(pool[i] == "A" for i in first) == 16


# ---------------------------------------------------------------- ECGDeli marks as waves

BEAT = {"P_on": 100, "P_peak": 120, "P_off": 140, "QRS_on": 160, "R": 170, "QRS_off": 190, "T_on": 230, "T_peak": 280, "T_off": 320}


def _points(n_beats: int = 3, step: int = 500, **changes: tuple[int, int]) -> dict[str, list[int]]:
    """n_beats copies of BEAT, ``step`` samples apart; ``changes``: point -> (beat, new value)."""
    points = {p: [v + k * step for k in range(n_beats)] for p, v in BEAT.items()}
    for p, (k, value) in changes.items():
        points[p][k] = value
    return points


def test_clean_beats_become_complete_waves() -> None:
    lw = ecgdeli_waves(_points(), 5000, af=False)
    assert lw.aligned and lw.n_beats == 3 and lw.n_masked_beats == 0
    assert len(lw.waves) == 9 and all(w.onset is not None and w.offset is not None for w in lw.waves)
    labels = labels_for(lw.waves, 5000)
    assert labels.classes[130] == CLASSES.index("P") and labels.classes[175] == CLASSES.index("QRS")
    assert labels.mask[100:1321].all()


def test_overlapping_beat_is_masked_in_that_lead() -> None:
    lw = ecgdeli_waves(_points(P_off=(1, 665)), 5000, af=False)  # beat 1's P ends after its QRS starts (660)
    assert lw.n_masked_beats == 1
    labels = labels_for(lw.waves, 5000)
    assert not labels.mask[321:1100].any()  # from beat 0's T end to beat 2's P start
    assert labels.mask[100:321].all() and labels.mask[1100:1321].all()


def test_t_running_into_the_next_beat_masks_both_beats() -> None:
    lw = ecgdeli_waves(_points(T_off=(0, 620)), 5000, af=False)  # beat 0's T ends after beat 1's P starts (600)
    assert lw.n_masked_beats == 2
    labels = labels_for(lw.waves, 5000)
    assert not labels.mask[:1100].any() and labels.mask[1100:1321].all()


def test_placeholders_are_missing() -> None:
    lw = ecgdeli_waves(_points(P_on=(0, 1), T_on=(2, 4998), T_peak=(2, 4999), T_off=(2, 5000)), 5000, af=False)
    first_p = lw.waves[0]
    assert first_p.kind == "P" and first_p.onset is None and first_p.offset == 140
    assert [w.kind for w in lw.waves].count("T") == 2  # the last beat's T had only placeholders
    labels = labels_for(lw.waves, 5000)
    assert not labels.mask[:160].any()  # around the P without onset
    assert not labels.mask[1191:].any()  # nothing after the last real mark


def test_af_drops_p_marks_so_the_p_region_is_background() -> None:
    lw = ecgdeli_waves(_points(), 5000, af=True)
    assert {w.kind for w in lw.waves} == {"QRS", "T"} and lw.n_masked_beats == 0
    labels = labels_for(lw.waves, 5000)
    assert labels.mask[600:640].all() and (labels.classes[600:640] == CLASSES.index("none")).all()


def test_unequal_counts_leave_the_lead_unlabelled() -> None:
    points = _points()
    points["P_on"] = points["P_on"][1:]
    lw = ecgdeli_waves(points, 5000, af=False)
    assert not lw.aligned and lw.waves == []
    assert ecgdeli_waves({}, 5000, af=False).waves == []
