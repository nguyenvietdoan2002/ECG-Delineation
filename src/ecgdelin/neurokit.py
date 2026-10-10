"""The NeuroKit2 `prominence` baseline (Emrich et al. 2024), run one lead at a time as a user would.

Default cleaning (`ecg_clean`), default R-peak detection (`ecg_peaks`), then
`ecg_delineate(method="prominence")`. R peaks are reported as `ECG_R_Peaks`. An exception on a lead
marks that lead as failed with no predictions (decision 42). ``rpeak_method="emrich2023"`` (FastNVG,
needs the `ts2vg` package) reproduces the paper's own pipeline for the sanity check (decision 46).
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import LEADS
from ecgdelin.delineate import KEYS, LeadPoints, empty_points


class NeuroKitProminence:
    def __init__(self, rpeak_method: str = "neurokit") -> None:
        self.rpeak_method = rpeak_method
        self.name = "neurokit_prominence" if rpeak_method == "neurokit" else f"neurokit_prominence_{rpeak_method}"

    def delineate(self, signal: NDArray[np.float64], fs: int) -> dict[str, LeadPoints]:
        return {lead: self.delineate_lead(signal[i], fs) for i, lead in enumerate(LEADS)}

    def delineate_lead(self, x: NDArray[np.float64], fs: int) -> LeadPoints:
        import neurokit2 as nk

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cleaned = nk.ecg_clean(x, sampling_rate=fs)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=self.rpeak_method)
                rpeaks = [int(r) for r in info["ECG_R_Peaks"]]
                if not rpeaks:
                    return empty_points()
                _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="prominence")
        except Exception:
            return empty_points(failed=True)
        points = {key: _per_beat(waves.get(key, []), len(rpeaks)) for key in KEYS if key != "ECG_R_Peaks"}
        points["ECG_R_Peaks"] = [float(r) for r in rpeaks]
        return LeadPoints(points)


def _per_beat(values: Sequence[float], n_beats: int) -> list[float]:
    """One float per beat, NaN where NeuroKit2 gives none; padded or cut to the number of R peaks."""
    out = [float(v) if v is not None and math.isfinite(float(v)) else math.nan for v in values][:n_beats]
    return out + [math.nan] * (n_beats - len(out))
