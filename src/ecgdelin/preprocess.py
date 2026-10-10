"""Preprocessing for the trained models: a 0.5 Hz zero-phase high-pass and the crop to 4992 samples.

Nothing else is done to the signal (plan, "Data"); NeuroKit2 uses its own cleaning on the full 5000.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

CUTOFF_HZ = 0.5
ORDER = 5
N_SAMPLES = 4992  # 156 blocks of 32; the latest LUDB mark is at sample 4832


PAD_SECONDS = 6.0  # scipy's default padding is far shorter than a 0.5 Hz filter's memory


def highpass(signal: NDArray[np.floating], fs: int) -> NDArray[np.float64]:
    """Zero-phase Butterworth high-pass along the last axis.

    Edge transients are unavoidable with any finite recording; mirrored ("even") padding of 6 s kept
    them smallest in a probe (about 0.5% of a 10 Hz wave's amplitude 1 s from the ends, 0.2% at 2 s).
    """
    x = np.asarray(signal, dtype=np.float64)
    sos = butter(ORDER, CUTOFF_HZ, btype="highpass", fs=fs, output="sos")
    padlen = min(int(PAD_SECONDS * fs), x.shape[-1] - 1)
    return np.asarray(sosfiltfilt(sos, x, axis=-1, padtype="even", padlen=padlen))


def prepare(signal: NDArray[np.floating], fs: int) -> NDArray[np.float32]:
    """(leads, samples) in mV -> (leads, 4992) float32, high-passed then cropped."""
    return highpass(signal, fs)[:, :N_SAMPLES].astype(np.float32)
