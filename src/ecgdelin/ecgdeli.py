"""ECGdeli's output in step 1's format (decision 78).

ECGdeli (Pilia et al., SoftwareX 13:100639, 2021) runs in MATLAB. ``Annotate_ECG_Multi`` returns, for
every lead, a fiducial point table (FPT): one row per beat, 13 columns of 1-based MATLAB sample
indices, 0 where a point was not found: P onset, P peak, P offset, QRS onset, Q peak, R peak, S peak,
QRS offset, L point, T onset, T peak, T offset, beat class. This module turns those tables into the
NeuroKit2-style ``LeadPoints`` the scorer reads: one beat per row, 0-based indices, NaN where absent.
"""

import math

import numpy as np
from numpy.typing import NDArray

from ecgdelin.delineate import KEYS, LeadPoints

# The key each FPT column feeds; None for the L point (column 9). Column 13, the beat class, is not read.
FPT_KEYS: tuple[str | None, ...] = (
    "ECG_P_Onsets",
    "ECG_P_Peaks",
    "ECG_P_Offsets",
    "ECG_R_Onsets",
    "ECG_Q_Peaks",
    "ECG_R_Peaks",
    "ECG_S_Peaks",
    "ECG_R_Offsets",
    None,
    "ECG_T_Onsets",
    "ECG_T_Peaks",
    "ECG_T_Offsets",
)


def lead_points_from_fpt(fpt: NDArray[np.float64], n_samples: int) -> LeadPoints:
    """One lead's FPT (beats x 13, 1-based) -> ``LeadPoints`` (0-based). Values outside 1..n_samples,
    including ECGdeli's 0 for a missing point, become NaN."""
    table = np.asarray(fpt, dtype=float)
    if table.size == 0:
        return LeadPoints({key: [] for key in KEYS})
    table = table.reshape(-1, table.shape[-1])
    if table.shape[1] < len(FPT_KEYS):
        raise ValueError(f"an FPT needs at least {len(FPT_KEYS)} columns, got {table.shape[1]}")
    points: dict[str, list[float]] = {key: [] for key in KEYS}
    for row in table:
        for column, key in enumerate(FPT_KEYS):
            if key is None:
                continue
            value = row[column]
            points[key].append(value - 1.0 if math.isfinite(value) and 1 <= value <= n_samples else math.nan)
    return LeadPoints(points)


def failed_lead() -> LeadPoints:
    """The output for a lead of a record ECGdeli raised on."""
    return LeadPoints({key: [] for key in KEYS}, failed=True)
