"""The scorer seam: predicted NeuroKit2-style points against LUDB marks, hand-built cases."""

import json

import numpy as np
import pytest

from ecgdelin.dataset import LEADS, Recording, Wave
from ecgdelin.delineate import POINT_TYPES, LeadPoints, empty_points, points_from_waves
from ecgdelin.score import compare, score

FS = 500


def beats(r_samples: list[int]) -> list[Wave]:
    """One P, QRS and T per beat at fixed offsets from each R sample."""
    waves = []
    for r in r_samples:
        waves += [
            Wave("P", r - 100, r - 80, r - 60),
            Wave("QRS", r - 20, r, r + 20),
            Wave("T", r + 100, r + 140, r + 180),
        ]
    return waves


def recording(rec_id: str, waves_by_lead: dict[str, list[Wave]], rhythm: str = "Sinus rhythm") -> Recording:
    marks = {lead: list(waves_by_lead.get(lead, [])) for lead in LEADS}
    return Recording(rec_id, FS, np.zeros((12, 5000)), rhythm, marks)


def shifted(waves: list[Wave], by: int) -> list[Wave]:
    return [
        Wave(w.kind, None if w.onset is None else w.onset + by, w.peak + by, None if w.offset is None else w.offset + by)
        for w in waves
    ]


def jitter(waves: list[Wave], by_beat: list[int]) -> list[Wave]:
    """Shift every wave of beat k (three consecutive waves) by ``by_beat[k]`` samples."""
    return [w for k in range(len(waves) // 3) for w in shifted(waves[3 * k : 3 * k + 3], by_beat[k])]


def predictions(**leads: list[Wave]) -> dict[str, LeadPoints]:
    return {lead: points_from_waves(waves) for lead, waves in leads.items()}


R_SAMPLES = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]


def test_perfect_prediction_scores_one_everywhere() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth})
    report = score([rec], {"1": predictions(ii=truth)})

    assert report.n_people == 1
    for point in POINT_TYPES:
        for tol in (150, 40):
            at = report.points[point].at[tol]
            assert (at.tp, at.fn, at.fp) == (9, 0, 0)
            assert (at.se, at.ppv, at.f1) == (1.0, 1.0, 1.0)
        assert report.points[point].mean_ms == 0.0
        assert report.points[point].sd_ms == 0.0
    assert report.primary_sd_ms == 0.0


def test_constant_10_ms_shift_gives_mean_10_and_sd_0() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth})
    report = score([rec], {"1": predictions(ii=shifted(truth, 5))})  # 5 samples = 10 ms late

    for point in POINT_TYPES:
        assert report.points[point].at[150].tp == 9
        assert report.points[point].at[40].tp == 9
        assert report.points[point].mean_ms == pytest.approx(10.0)
        assert report.points[point].sd_ms == 0.0
    assert report.primary_sd_ms == 0.0


def test_a_missed_beat_is_a_fn_and_an_extra_beat_is_a_fp() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth})
    pred = beats([r for r in R_SAMPLES if r != 3000] + [2250])  # beat at 3000 missed, extra beat at 2250
    report = score([rec], {"1": predictions(ii=pred)})

    for point in POINT_TYPES:
        at = report.points[point].at[150]
        assert (at.tp, at.fn, at.fp) == (8, 1, 1)
        assert at.se == pytest.approx(8 / 9)
        assert at.ppv == pytest.approx(8 / 9)
        assert at.f1 == pytest.approx(16 / 18)


def test_tolerance_window_is_inclusive_and_strict_window_is_separate() -> None:
    truth = beats([500, 2500, 4500])
    rec = recording("1", {"ii": truth})

    late_152 = beats([500, 2500 + 76, 4500])  # middle beat 152 ms late: outside +/-150 ms
    report = score([rec], {"1": predictions(ii=late_152)})
    for point in POINT_TYPES:
        assert (report.points[point].at[150].tp, report.points[point].at[150].fn, report.points[point].at[150].fp) == (2, 1, 1)

    late_150 = beats([500, 2500 + 75, 4500])  # exactly 150 ms late: inside the loose window only
    report = score([rec], {"1": predictions(ii=late_150)})
    for point in POINT_TYPES:
        assert (report.points[point].at[150].tp, report.points[point].at[150].fn, report.points[point].at[150].fp) == (3, 0, 0)
        assert (report.points[point].at[40].tp, report.points[point].at[40].fn, report.points[point].at[40].fp) == (2, 1, 1)
        assert report.points[point].mean_ms == pytest.approx(50.0)


def test_predictions_outside_the_marked_span_are_ignored() -> None:
    truth = beats([1000, 2000, 3000])  # marked span: samples 900..3180
    rec = recording("1", {"ii": truth})
    report = score([rec], {"1": predictions(ii=beats([200, 1000, 2000, 3000, 4800]))})

    for point in POINT_TYPES:
        at = report.points[point].at[150]
        assert (at.tp, at.fn, at.fp) == (3, 0, 0)


def test_the_gap_around_an_incomplete_wave_is_ignored_on_both_sides() -> None:
    truth = beats([1000, 2000, 3000])
    truth[5] = Wave("T", 2100, 2140, None)  # the T of the middle beat has no offset
    rec = recording("1", {"ii": truth})
    report = score([rec], {"1": predictions(ii=beats([1000, 2000, 3000]))})  # predicts a full T there

    for point in ("T_on", "T_peak", "T_off"):
        at = report.points[point].at[150]
        assert (at.tp, at.fn, at.fp) == (2, 0, 0)
    for point in ("P_on", "QRS_peak", "P_off"):
        assert report.points[point].at[150].tp == 3


def test_within_recording_sd_averages_per_lead_sds_while_the_pooled_sd_mixes_them() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth, "v1": truth})
    report = score([rec], {"1": predictions(ii=shifted(truth, 5), v1=shifted(truth, -5))})

    for point in POINT_TYPES:
        assert report.points[point].mean_ms == pytest.approx(0.0)
        assert report.points[point].sd_ms == pytest.approx(10.0)  # population SD of nine +10 and nine -10
        assert report.points[point].sd_within_ms == 0.0  # each lead alone has a constant error


def test_region_overlap_is_perfect_for_exact_regions_and_drops_for_widened_qrs() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth})

    exact = score([rec], {"1": predictions(ii=truth)})
    assert exact.miou == {"P": 1.0, "QRS": 1.0, "T": 1.0, "mean": 1.0}

    wide = [Wave(w.kind, w.onset - 10, w.peak, w.offset + 10) if w.kind == "QRS" else w for w in truth]
    report = score([rec], {"1": predictions(ii=wide)})
    assert report.miou["QRS"] == pytest.approx(41 / 61)  # 41 true samples inside 61 predicted ones
    assert report.miou["P"] == 1.0
    assert report.miou["T"] == 1.0
    assert report.miou["mean"] == pytest.approx((2 + 41 / 61) / 3)


def test_metrics_are_broken_down_by_rhythm_group() -> None:
    truth = beats(R_SAMPLES)
    sinus = recording("1", {"ii": truth}, rhythm="Sinus rhythm")
    af = recording("2", {"ii": truth}, rhythm="Atrial fibrillation")
    report = score([sinus, af], {"1": predictions(ii=truth), "2": predictions(ii=shifted(truth, 5))})

    assert set(report.by_rhythm) == {"sinus", "af_flutter"}
    assert report.by_rhythm["sinus"].n_people == 1
    assert report.by_rhythm["sinus"].points["P_on"].mean_ms == 0.0
    assert report.by_rhythm["af_flutter"].points["P_on"].mean_ms == pytest.approx(10.0)
    assert report.points["P_on"].mean_ms == pytest.approx(5.0)


EIGHT = R_SAMPLES[:8]  # an even number of beats, so alternating errors sum to zero


def test_bootstrap_interval_brackets_the_estimate_and_collapses_for_identical_people() -> None:
    truth = beats(EIGHT)
    recs = [recording(str(i), {"ii": truth}) for i in (1, 2, 3)]
    same = jitter(truth, [2, -2] * 4)  # every person: errors of +4 and -4 ms, SD 4 ms

    report = score(recs, {rec.id: predictions(ii=same) for rec in recs}, n_draws=2000, seed=0)
    assert report.primary_sd_ms == pytest.approx(4.0)
    assert report.primary_ci95 == pytest.approx((4.0, 4.0))

    mixed = {"1": predictions(ii=same), "2": predictions(ii=jitter(truth, [10, -10] * 4)), "3": predictions(ii=truth)}
    report = score(recs, mixed, n_draws=2000, seed=0)
    lo, hi = report.primary_ci95
    assert lo < report.primary_sd_ms < hi
    assert lo >= 0.0


def test_stop_rule_says_yes_only_when_the_paired_difference_clears_zero() -> None:
    truth = beats(EIGHT)
    recs = [recording(str(i), {"ii": truth}) for i in (1, 2, 3)]
    tight = predictions(ii=jitter(truth, [1, -1] * 4))  # +/-2 ms
    loose = predictions(ii=jitter(truth, [20, -20] * 4))  # +/-40 ms, inside both windows

    verdict = compare(recs, {r.id: tight for r in recs}, {r.id: loose for r in recs}, n_draws=2000, seed=0)
    assert verdict.diff_ci95[1] < 0  # a minus b, lower SD is better
    assert verdict.a_better is True
    assert verdict.f1_safeguard_ok is True
    assert verdict.a_wins is True

    # Roles swap across people: the interval of the difference straddles zero.
    a = {"1": tight, "2": loose, "3": tight}
    b = {"1": loose, "2": tight, "3": loose}
    verdict = compare(recs, a, b, n_draws=2000, seed=0)
    assert verdict.diff_ci95[0] < 0 < verdict.diff_ci95[1]
    assert verdict.a_better is False
    assert verdict.a_wins is False

    # Tighter errors but a missed beat on every person: better SD, worse F1, so no win.
    missing = predictions(ii=jitter(truth[:-3], [1, -1] * 4))
    verdict = compare(recs, {r.id: missing for r in recs}, {r.id: loose for r in recs}, n_draws=2000, seed=0)
    assert verdict.a_better is True
    assert verdict.f1_diff_ci95[1] < 0
    assert verdict.f1_safeguard_ok is False
    assert verdict.a_wins is False


def test_failed_leads_and_orphans_are_counted_and_failed_leads_count_as_misses() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth, "v1": truth})
    pred = {
        "ii": points_from_waves(truth + [Wave("P", 4700, 4720, 4740)]),  # a P after the last QRS
        "v1": empty_points(failed=True),
    }
    report = score([rec], {"1": pred})

    assert report.orphans == 1
    assert report.failed_leads == 1
    assert (report.points["P_on"].at[150].tp, report.points["P_on"].at[150].fn, report.points["P_on"].at[150].fp) == (9, 9, 0)


def test_report_serialises_to_json_without_nan() -> None:
    truth = beats(R_SAMPLES)
    rec = recording("1", {"ii": truth})

    back = json.loads(json.dumps(score([rec], {"1": predictions(ii=truth)}).to_dict(), allow_nan=False))
    assert back["primary_sd_ms"] == 0.0
    assert back["points"]["P_on"]["at"]["150"]["f1"] == 1.0
    assert back["points"]["P_on"]["cse_2sd_ms"] == 10.2
    assert back["by_rhythm"]["sinus"]["n_people"] == 1

    nothing = json.loads(json.dumps(score([rec], {}).to_dict(), allow_nan=False))
    assert nothing["points"]["P_on"]["mean_ms"] is None
    assert nothing["points"]["P_on"]["at"]["150"]["se"] == 0.0
