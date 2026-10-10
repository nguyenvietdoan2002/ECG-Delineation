"""Preprocessing for the trained models: 0.5 Hz zero-phase high-pass, crop to 4992 samples."""

import numpy as np

from ecgdelin.preprocess import N_SAMPLES, highpass, prepare

FS = 500
MIDDLE = slice(1000, 4000)  # 2 s from either end, away from the filter's edge transients


def test_highpass_removes_offset_and_slow_drift() -> None:
    t = np.arange(5000) / FS
    signal = np.vstack([np.full(5000, 1.0), np.sin(2 * np.pi * 0.1 * t)])  # 1 mV offset; 0.1 Hz drift

    out = highpass(signal, FS)

    assert out.shape == signal.shape
    assert np.max(np.abs(out[0, MIDDLE])) < 1e-3
    assert np.max(np.abs(out[1, MIDDLE])) < 0.02


def test_highpass_passes_a_10_hz_wave_unchanged_and_without_delay_away_from_the_edges() -> None:
    t = np.arange(5000) / FS
    wave = np.sin(2 * np.pi * 10 * t)

    out = highpass(wave[None, :], FS)[0]

    np.testing.assert_allclose(out[MIDDLE], wave[MIDDLE], atol=3e-3)
    assert np.max(np.abs(out[500:4500] - wave[500:4500])) < 0.01  # within 1 s of the ends: under 1%


def test_prepare_filters_then_crops_to_4992_samples_as_float32() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(12, 5000)) + 3.0

    out = prepare(signal, FS)

    assert out.shape == (12, N_SAMPLES)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, highpass(signal, FS)[:, :N_SAMPLES].astype(np.float32))
