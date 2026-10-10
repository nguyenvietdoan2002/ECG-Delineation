"""End-to-end on real LUDB records: load, delineate with NeuroKit2, score. Outputs must be well-formed
and the scores in a plausible range (Emrich et al. 2024 report F1 above 98% and error SDs under 17 ms)."""

from pathlib import Path

import pytest

from ecgdelin.dataset import LEADS, load_record
from ecgdelin.delineate import CSE_POINTS, KEYS, POINT_TYPES
from ecgdelin.neurokit import NeuroKitProminence
from ecgdelin.score import score


@pytest.mark.data
@pytest.mark.slow
def test_neurokit_prominence_on_three_records_scores_in_a_plausible_range(ludb_root: Path) -> None:
    recordings = [load_record(ludb_root, rec_id) for rec_id in ("1", "2", "3")]
    method = NeuroKitProminence()
    predictions = {rec.id: method.delineate(rec.signal, rec.fs) for rec in recordings}

    for rec in recordings:
        for lead in LEADS:
            points = predictions[rec.id][lead]
            assert set(points.points) == set(KEYS)
            assert points.failed is False
            assert 5 <= points.n_beats <= 20

    report = score(recordings, predictions, n_draws=200)
    assert report.n_people == 3
    for point in POINT_TYPES:
        assert report.points[point].at[150].f1 > 0.8, point
        assert abs(report.points[point].mean_ms) < 40, point
    for point in CSE_POINTS:
        assert report.points[point].sd_ms < 40, point
    assert report.primary_sd_ms < 30
    assert report.miou["QRS"] > 0.6
    assert set(report.by_rhythm) <= {"sinus", "af_flutter", "other"}
