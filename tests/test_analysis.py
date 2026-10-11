"""Development analysis helpers for the M3 deep dive (decision 60): per-lead errors, the orphan
breakdown of a raw class map, the marks round trip through the decoder, and a person's primary SD."""

import numpy as np

from ecgdelin.analysis import orphan_breakdown, person_primary_sd, roundtrip_points
from ecgdelin.dataset import LEADS, Recording, Wave
from ecgdelin.decode import regions_to_points
from ecgdelin.delineate import POINT_TYPES, points_from_waves
from ecgdelin.labels import CLASSES
from ecgdelin.score import errors_by_lead, score_recording

FS = 500
N = 5000
P, QRS, T = (CLASSES.index(k) for k in ("P", "QRS", "T"))


def beats(r_samples: list[int]) -> list[Wave]:
    waves = []
    for r in r_samples:
        waves += [Wave("P", r - 100, r - 80, r - 60), Wave("QRS", r - 20, r, r + 20), Wave("T", r + 100, r + 140, r + 180)]
    return waves


def shifted(waves: list[Wave], by: int) -> list[Wave]:
    return [Wave(w.kind, None if w.onset is None else w.onset + by, w.peak + by, None if w.offset is None else w.offset + by) for w in waves]


def recording(waves_by_lead: dict[str, list[Wave]], signal: np.ndarray | None = None) -> Recording:
    marks = {lead: list(waves_by_lead.get(lead, [])) for lead in LEADS}
    return Recording("1", FS, np.zeros((12, N)) if signal is None else signal, "Sinus rhythm", marks)


def bump(centre: int, amplitude: float, width: float = 4.0) -> np.ndarray:
    t = np.arange(N)
    return amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)


def test_errors_by_lead_keeps_each_lead_apart() -> None:
    waves = beats([1000, 2000, 3000])
    rec = recording({"i": waves, "ii": waves})
    pred = {"i": points_from_waves(waves), "ii": points_from_waves(shifted(waves, 5))}

    out = errors_by_lead(rec, pred)

    for point in POINT_TYPES:
        assert np.all(out["i"].errors_ms[point] == 0.0)
        assert np.all(out["ii"].errors_ms[point] == 10.0)  # 5 samples at 500 Hz
        assert out["iii"].errors_ms[point].size == 0
    assert (out["i"].counts["P_on"].tp, out["i"].counts["P_on"].fn, out["i"].counts["P_on"].fp) == (3, 0, 0)
    assert out["iii"].counts["QRS_on"].tp == 0
    # The same matching as the whole-recording scorer: its pooled errors are the leads' errors joined.
    pooled = score_recording(rec, pred)
    assert sorted(pooled.errors_ms["T_off"].tolist()) == sorted(out["i"].errors_ms["T_off"].tolist() + out["ii"].errors_ms["T_off"].tolist())


def test_orphan_breakdown_separates_edge_extra_and_unanchored() -> None:
    waves = beats([1300, 2300])  # marks span sample 1200 (first P onset) to 2480 (last T offset)
    rec = recording({"i": waves})
    classes = np.zeros((12, N), dtype=np.int8)
    lead = classes[0]
    lead[100:161] = T  # before the first mark: edge
    lead[1200:1241] = P
    lead[1280:1321] = QRS
    lead[1400:1481] = T
    lead[1700:1741] = P  # a second P before the next QRS, whose slot the nearer P takes: extra
    lead[2200:2241] = P
    lead[2280:2321] = QRS
    lead[2400:2431] = T
    lead[2450:2476] = P  # inside the span, after the last predicted QRS: unanchored
    lead[4800:4851] = P  # after the last mark: edge

    out = orphan_breakdown(classes, rec, rec.signal)

    assert out == {"edge": 2, "extra": 1, "unanchored": 1}
    assert sum(out.values()) == sum(regions_to_points(classes[k], rec.signal[k], FS).orphans for k in range(12))


def test_roundtrip_points_reproduce_the_marks() -> None:
    waves = beats([1000, 2000, 3000])
    signal = np.zeros((12, N))
    signal[0] = sum(bump(w.peak, {"P": 0.2, "QRS": 1.0, "T": 0.3}[w.kind]) for w in waves)
    rec = recording({"i": waves}, signal)

    points = roundtrip_points(rec, signal)

    assert points["i"].points["ECG_R_Onsets"] == [980, 1980, 2980]
    assert points["i"].points["ECG_R_Peaks"] == [1000, 2000, 3000]
    assert points["i"].points["ECG_T_Offsets"] == [1180, 2180, 3180]
    assert points["i"].points["ECG_P_Peaks"] == [920, 1920, 2920]
    assert points["ii"].n_beats == 0
    for point in POINT_TYPES:
        assert np.all(errors_by_lead(rec, points)["i"].errors_ms[point] == 0.0)


def test_person_primary_sd_is_the_mean_error_sd_over_the_cse_points() -> None:
    waves = beats([1000, 2000, 3000])
    rec = recording({"i": waves})
    jittered = [Wave(w.kind, w.onset + by, w.peak + by, w.offset + by) for w, by in zip(waves, [0] * 3 + [5] * 3 + [10] * 3)]

    person = score_recording(rec, {"i": points_from_waves(jittered)})

    assert abs(person_primary_sd(person) - float(np.std([0.0, 10.0, 20.0]))) < 1e-9
    assert np.isnan(person_primary_sd(score_recording(rec, {})))
