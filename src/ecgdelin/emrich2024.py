"""Published LUDB numbers for NeuroKit2's `prominence` method (Emrich et al., EUSIPCO 2024, Table II).

Single-lead setting, first LUDB lead (lead I), 150 ms tolerance: F1 in %, mean error and the average
of per-recording error SDs in ms. See docs/references.md for the protocol and sources.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

from ecgdelin.score import Report

F1_BAND_POINTS = 1.0  # decision 43: F1 within 1 percentage point ...
SD_BAND_MS = 3.0  # ... and each error SD within 3 ms


class Published(NamedTuple):
    f1_pct: float
    mean_ms: float
    sd_within_ms: float


SINGLE_LEAD_I: Mapping[str, Published] = {
    "P_on": Published(99.75, 10.6, 11.6),
    "P_peak": Published(99.82, 2.6, 5.8),
    "P_off": Published(99.82, 6.8, 9.0),
    "QRS_on": Published(99.81, 5.8, 6.2),
    "QRS_peak": Published(99.81, 1.6, 2.0),
    "QRS_off": Published(99.81, -9.2, 7.2),
    "T_on": Published(98.30, 5.6, 16.6),
    "T_peak": Published(98.33, 2.6, 8.6),
    "T_off": Published(98.54, 14.2, 16.0),
}

TWELVE_LEAD_WITH_CORRECTION: Mapping[str, Published] = {
    "P_on": Published(99.58, 1.2, 4.8),
    "P_peak": Published(99.54, 0.6, 3.8),
    "P_off": Published(99.56, 0.6, 4.0),
    "QRS_on": Published(99.92, -0.2, 2.0),
    "QRS_peak": Published(99.97, -0.2, 1.8),
    "QRS_off": Published(99.94, -6.0, 4.0),
    "T_on": Published(99.19, 0.6, 6.6),
    "T_peak": Published(99.15, 0.2, 4.0),
    "T_off": Published(99.22, 2.4, 7.6),
}


def compare_to_published(lead_i_report: Report, published: Mapping[str, Published] = SINGLE_LEAD_I) -> list[dict[str, Any]]:
    """Our lead-I F1 at 150 ms, mean and within-recording SD next to the paper's, with the decision-43 band."""
    rows = []
    for point, theirs in published.items():
        ours = lead_i_report.points[point]
        f1 = ours.at[150].f1 * 100
        rows.append(
            {
                "point": point,
                "f1_ours": round(f1, 2),
                "f1_paper": theirs.f1_pct,
                "mean_ours": round(ours.mean_ms, 1),
                "mean_paper": theirs.mean_ms,
                "sd_within_ours": round(ours.sd_within_ms, 1),
                "sd_within_paper": theirs.sd_within_ms,
                "within_band": abs(f1 - theirs.f1_pct) <= F1_BAND_POINTS and abs(ours.sd_within_ms - theirs.sd_within_ms) <= SD_BAND_MS,
            }
        )
    return rows
