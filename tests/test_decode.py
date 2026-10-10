"""The decoder: per-sample regions -> NeuroKit2-style points, and the marks -> labels -> points round trip."""

import math
from pathlib import Path

import numpy as np
import pytest

from ecgdelin.dataset import LEADS, load_record
from ecgdelin.decode import regions_to_points
from ecgdelin.delineate import points_from_waves
from ecgdelin.labels import CLASSES, labels_for

FS = 500
P, QRS, T = (CLASSES.index(k) for k in ("P", "QRS", "T"))


def bump(n: int, centre: int, amplitude: float, width: float = 4.0) -> np.ndarray:
    t = np.arange(n)
    return amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)


def test_regions_become_beats_with_peaks_from_the_signal_after_clean_up() -> None:
    n = 1000
    classes = np.zeros(n, dtype=np.int8)
    classes[100:141] = P  # beat 1
    classes[200:241] = QRS
    classes[300:381] = T
    classes[500:506] = P  # 6 samples = 12 ms: dropped (< 20 ms)
    classes[800:841] = QRS  # beat 2, no P
    classes[900:941] = T  # 2-sample gap (4 ms) joins the two T pieces
    classes[943:1000] = T
    signal = (
        bump(n, 120, 0.2)  # P peak
        + bump(n, 210, -0.1, 2) + bump(n, 220, 1.0, 2) + bump(n, 230, -0.3, 2)  # Q, R, S
        + bump(n, 340, -0.3, 10)  # inverted T: largest absolute deflection
        + bump(n, 812, -0.1, 2) + bump(n, 820, 1.0, 2) + bump(n, 828, -0.2, 2)
        + bump(n, 950, 0.3, 10)
    )

    out = regions_to_points(classes, signal, FS)

    expected = {
        "ECG_P_Onsets": [100, math.nan],
        "ECG_P_Peaks": [120, math.nan],
        "ECG_P_Offsets": [140, math.nan],
        "ECG_R_Onsets": [200, 800],
        "ECG_Q_Peaks": [210, 812],
        "ECG_R_Peaks": [220, 820],
        "ECG_S_Peaks": [230, 828],
        "ECG_R_Offsets": [240, 840],
        "ECG_T_Onsets": [300, 900],
        "ECG_T_Peaks": [340, 950],
        "ECG_T_Offsets": [380, 999],
    }
    assert set(out.points) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(out.points[key], values, err_msg=key)
    assert out.orphans == 0
    assert out.failed is False


def test_a_t_wave_region_shorter_than_40_ms_is_dropped_and_a_p_after_the_last_qrs_is_an_orphan() -> None:
    n = 1000
    classes = np.zeros(n, dtype=np.int8)
    classes[200:241] = QRS
    classes[300:318] = T  # 18 samples = 36 ms: dropped (< 40 ms)
    classes[600:641] = P  # no QRS after it: orphan
    signal = bump(n, 220, 1.0, 2) + bump(n, 309, 0.2) + bump(n, 620, 0.2)

    out = regions_to_points(classes, signal, FS)

    assert out.points["ECG_R_Peaks"] == [220]
    assert math.isnan(out.points["ECG_T_Onsets"][0])
    assert math.isnan(out.points["ECG_P_Onsets"][0])
    assert out.orphans == 1


@pytest.mark.data
def test_ludb_marks_survive_the_round_trip_through_labels_and_the_decoder(ludb_root: Path) -> None:
    """Marks -> per-sample labels -> regions -> points must give back every complete wave's onset
    and offset exactly; peaks come from the signal, so they are checked loosely against the marks."""
    peak_keys = ("ECG_P_Peaks", "ECG_R_Peaks", "ECG_T_Peaks")
    peak_errors: dict[str, list[float]] = {key: [] for key in peak_keys}
    for rec_id in ("1", "2", "3", "4", "5"):
        rec = load_record(ludb_root, rec_id)
        for lead in LEADS:
            waves = rec.marks[lead]
            complete = [w for w in waves if w.onset is not None and w.offset is not None]
            expected = points_from_waves(complete)
            got = regions_to_points(labels_for(waves, rec.signal.shape[1]).classes, rec.lead(lead), rec.fs)
            for key in ("ECG_P_Onsets", "ECG_P_Offsets", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Offsets"):
                np.testing.assert_array_equal(got.points[key], expected.points[key], err_msg=f"{rec_id} {lead} {key}")
            for key in peak_keys:
                g, e = np.array(got.points[key]), np.array(expected.points[key])
                both = np.isfinite(g) & np.isfinite(e)
                peak_errors[key] += list(np.abs(g[both] - e[both]))
    for key in peak_keys:
        errors = np.array(peak_errors[key])
        assert errors.size > 150, key
        assert np.median(errors) <= 2, key  # samples (4 ms)
        assert np.mean(errors <= 10) >= 0.93, key  # within 20 ms
