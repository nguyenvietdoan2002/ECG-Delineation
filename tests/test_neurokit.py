"""The delineator seam for the NeuroKit2 `prominence` baseline, on synthetic ECGs with known waves."""

import math

import numpy as np
from synthetic import FS, synthetic_ecg

from ecgdelin.dataset import LEADS
from ecgdelin.delineate import KEYS
from ecgdelin.neurokit import NeuroKitProminence


def finite(values: list[float]) -> np.ndarray:
    return np.array([v for v in values if math.isfinite(v)])


def test_every_lead_gets_the_eleven_keys_with_one_value_per_beat_inside_the_signal() -> None:
    ecg = synthetic_ecg()
    out = NeuroKitProminence().delineate(ecg.signal, FS)

    assert set(out) == set(LEADS)
    for lead in LEADS:
        points = out[lead].points
        assert set(points) == set(KEYS)
        n_beats = len(points["ECG_R_Peaks"])
        assert n_beats >= len(ecg.waves["R"]) - 2, lead
        for key in KEYS:
            assert len(points[key]) == n_beats, (lead, key)
            values = finite(points[key])
            assert np.all((values >= 0) & (values < ecg.signal.shape[1])), (lead, key)
        assert out[lead].failed is False


def test_r_peaks_land_on_the_true_r_waves_and_waves_are_ordered_within_each_beat() -> None:
    ecg = synthetic_ecg()
    points = NeuroKitProminence().delineate(ecg.signal, FS)["ii"].points

    r = np.array(points["ECG_R_Peaks"])
    nearest_true = ecg.waves["R"][np.abs(ecg.waves["R"][None, :] - r[:, None]).argmin(axis=1)]
    np.testing.assert_allclose(r, nearest_true, atol=2)

    for k in range(len(r)):
        p = [points[key][k] for key in ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets")]
        q = [points[key][k] for key in ("ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets")]
        t = [points[key][k] for key in ("ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")]
        if all(map(math.isfinite, p)):
            assert p[0] <= p[1] <= p[2] < q[1], k
        assert q[0] <= q[1] <= q[2], k
        if all(map(math.isfinite, t)):
            assert q[1] < t[0] <= t[1] <= t[2], k


def test_p_peaks_match_the_synthetic_p_waves_within_10_ms_in_lead_ii() -> None:
    ecg = synthetic_ecg()
    points = NeuroKitProminence().delineate(ecg.signal, FS)["ii"].points

    p = finite(points["ECG_P_Peaks"])
    assert p.size >= len(ecg.waves["P"]) - 2
    nearest_true = ecg.waves["P"][np.abs(ecg.waves["P"][None, :] - p[:, None]).argmin(axis=1)]
    np.testing.assert_allclose(p, nearest_true, atol=0.010 * FS)


def test_without_p_waves_neurokit_still_reports_p_points_on_noise() -> None:
    """A known property of `prominence`: it picks the most prominent bump in the 300 ms before Q
    whatever its size, so an AF-like signal gets P points the scorer will count as false positives."""
    ecg = synthetic_ecg(rr=[0.75, 1.1, 0.85, 0.95], p_waves=False, noise_mv=0.005)
    points = NeuroKitProminence().delineate(ecg.signal, FS)["ii"].points

    assert len(points["ECG_R_Peaks"]) >= len(ecg.waves["R"]) - 2
    p = finite(points["ECG_P_Peaks"])
    assert p.size >= 0.8 * len(points["ECG_R_Peaks"])
    assert np.all(np.abs(ecg.signal[1, p.astype(int)]) < 0.03)  # those "P peaks" are noise-sized


def test_a_flat_lead_yields_nothing_and_does_not_break_the_other_leads() -> None:
    ecg = synthetic_ecg()
    signal = ecg.signal.copy()
    signal[LEADS.index("v6")] = 0.0
    out = NeuroKitProminence().delineate(signal, FS)

    assert out["v6"].failed or out["v6"].n_beats == 0
    assert out["ii"].n_beats >= len(ecg.waves["R"]) - 2
