"""Pooling the folds' test predictions into one prediction set (decision 64): every person exactly once."""

import numpy as np
import pytest

from ecgdelin.crossval import assemble, per_fold_primary
from ecgdelin.dataset import LEADS, Recording, Wave
from ecgdelin.delineate import points_from_waves
from ecgdelin.reporting import fmt, md_table
from ecgdelin.splits import Fold, Folds

FS = 500


def beats(r_samples: list[int]) -> list[Wave]:
    waves = []
    for r in r_samples:
        waves += [Wave("P", r - 100, r - 80, r - 60), Wave("QRS", r - 20, r, r + 20), Wave("T", r + 100, r + 140, r + 180)]
    return waves


def recording(rec_id: str) -> Recording:
    waves = beats([1000, 2000, 3000])
    return Recording(rec_id, FS, np.zeros((12, 5000)), "Sinus rhythm", {lead: list(waves) for lead in LEADS})


def shifted(waves: list[Wave], by: int) -> list[Wave]:
    return [Wave(w.kind, None if w.onset is None else w.onset + by, w.peak + by, None if w.offset is None else w.offset + by) for w in waves]


FOLDS = Folds(seed=0, folds=(Fold(test=("1", "2"), val=("3",), train=("4",), subsets={}), Fold(test=("3", "4"), val=("1",), train=("2",), subsets={})))


def predictions(ids: list[str], shift: int) -> dict:
    return {i: {lead: points_from_waves(shifted(beats([1000, 2000, 3000]), shift)) for lead in LEADS} for i in ids}


def test_assemble_takes_each_person_from_the_fold_that_held_them_out() -> None:
    out = assemble(FOLDS, {0: predictions(["1", "2"], 0), 1: predictions(["3", "4"], 5)})

    assert sorted(out) == ["1", "2", "3", "4"]
    assert out["1"]["ii"].points["ECG_R_Peaks"] == [1000, 2000, 3000]
    assert out["3"]["ii"].points["ECG_R_Peaks"] == [1005, 2005, 3005]
    assert sorted(assemble(FOLDS, {1: predictions(["3", "4"], 5)}, only=[1])) == ["3", "4"]


def test_assemble_refuses_gaps_duplicates_and_strangers() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        assemble(FOLDS, {0: predictions(["1", "2"], 0)})
    with pytest.raises(ValueError, match="no prediction for test person 2"):
        assemble(FOLDS, {0: predictions(["1"], 0), 1: predictions(["3", "4"], 0)})
    with pytest.raises(ValueError, match="outside its test set"):
        assemble(FOLDS, {0: predictions(["1", "2", "3"], 0), 1: predictions(["3", "4"], 0)})


def test_per_fold_primary_averages_the_seeds_of_each_fold() -> None:
    recs = {i: recording(i) for i in ("1", "2", "3", "4")}
    jittered = {i: {lead: points_from_waves([Wave(w.kind, w.onset + by, w.peak + by, w.offset + by) for w, by in zip(beats([1000, 2000, 3000]), [0] * 3 + [5] * 3 + [10] * 3)]) for lead in LEADS} for i in ("3", "4")}

    out = per_fold_primary(recs, FOLDS, {0: [predictions(["1", "2"], 0), predictions(["1", "2"], 7)], 1: [jittered, predictions(["3", "4"], 0)]})

    assert out[0] == 0.0  # both seeds: a constant shift has SD 0
    assert abs(out[1] - float(np.std([0.0, 10.0, 20.0])) / 2) < 1e-9  # one jittered seed, one perfect seed


def test_markdown_helpers() -> None:
    assert fmt(float("nan")) == "–" and fmt(None) == "–" and fmt(3.14159, 2) == "3.14"
    assert md_table(["a", "b"], [[1, "x"]]) == "| a | b |\n|---|---|\n| 1 | x |"
