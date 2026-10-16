import math

import numpy as np
import pytest

from ecgdelin.delineate import KEYS
from ecgdelin.ecgdeli import failed_lead, lead_points_from_fpt


def _row(**marks: float) -> list[float]:
    """An FPT row with 1-based indices; columns not given are 0 (ECGdeli's 'not found')."""
    order = ["p_on", "p_pk", "p_off", "qrs_on", "q", "r", "s", "qrs_off", "l", "t_on", "t_pk", "t_off", "cls"]
    return [float(marks.get(name, 0)) for name in order]


def test_columns_map_to_neurokit_keys_and_become_zero_based() -> None:
    fpt = np.array([_row(p_on=101, p_pk=121, p_off=141, qrs_on=161, q=165, r=171, s=178, qrs_off=191, l=200,
                         t_on=231, t_pk=281, t_off=321, cls=1)])
    lp = lead_points_from_fpt(fpt, 5000)
    expected = {
        "ECG_P_Onsets": 100, "ECG_P_Peaks": 120, "ECG_P_Offsets": 140, "ECG_R_Onsets": 160, "ECG_Q_Peaks": 164,
        "ECG_R_Peaks": 170, "ECG_S_Peaks": 177, "ECG_R_Offsets": 190, "ECG_T_Onsets": 230, "ECG_T_Peaks": 280,
        "ECG_T_Offsets": 320,
    }
    assert set(lp.points) == set(KEYS)
    for key, value in expected.items():
        assert lp.points[key] == [value]
    assert lp.n_beats == 1 and not lp.failed


def test_zero_and_out_of_range_points_are_nan() -> None:
    fpt = np.array([_row(qrs_on=161, r=171, qrs_off=191, t_on=4998, t_pk=4999, t_off=5001)])
    lp = lead_points_from_fpt(fpt, 5000)
    assert math.isnan(lp.points["ECG_P_Onsets"][0])
    assert math.isnan(lp.points["ECG_P_Peaks"][0])
    assert lp.points["ECG_T_Peaks"] == [4998.0]
    assert math.isnan(lp.points["ECG_T_Offsets"][0])  # 5001 is past the last sample


def test_one_beat_per_row_and_empty_tables() -> None:
    fpt = np.array([_row(qrs_on=161, r=171, qrs_off=191), _row(qrs_on=661, r=671, qrs_off=691)])
    assert lead_points_from_fpt(fpt, 5000).points["ECG_R_Peaks"] == [170.0, 670.0]
    empty = lead_points_from_fpt(np.zeros((0, 13)), 5000)
    assert empty.n_beats == 0 and not empty.failed
    assert lead_points_from_fpt(np.array([]), 5000).n_beats == 0


def test_short_tables_are_refused_and_failures_are_marked() -> None:
    with pytest.raises(ValueError):
        lead_points_from_fpt(np.ones((2, 8)), 5000)
    assert failed_lead().failed and failed_lead().n_beats == 0
