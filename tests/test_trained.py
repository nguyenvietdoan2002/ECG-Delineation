"""Trained models behind the delineator seam (spec testing decisions).

A small head over a stub backbone and a small U-Net are trained on synthetic ECGs in a few seconds;
like every method they must return the eleven keys per lead, keep P before QRS before T, and give NaN
where a wave is absent. The fixture's synthetic set mixes recordings with and without P waves and uses
a tall P wave, so that both tiny models learn all three waves within a few hundred steps.
"""

import math
import os
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ecgdelin.dataset import LEADS
from ecgdelin.delineate import KEYS, Delineator
from ecgdelin.labels import labels_for
from ecgdelin.models import ConvHead, LinearProbe, UNet1D, count_parameters
from ecgdelin.preprocess import N_SAMPLES, prepare
from ecgdelin.states import extract_states
from ecgdelin.train import TensorSource, TrainConfig, make_trainee, train_jointly
from ecgdelin.trained import HeadDelineator, UNetDelineator
from synthetic import Synthetic, synthetic_ecg, synthetic_waves

P_MV = 0.4  # a tall P wave: the tiny U-Net learns each lead's P readout separately, which is slow with a small one
TRAIN = ((0.75, True), (0.85, False), (0.95, True), (1.05, False), (1.15, True), (1.25, False))  # (RR in s, has P waves)
VAL = ((1.0, True),)
WINDOW = 416  # the U-Net trains on 12 windows per recording: many cheap steps instead of few expensive ones


class RawPatchBackbone:
    """A one-layer 'backbone' whose state for a patch is the patch itself: 32 raw samples."""

    n_layers = 1
    d_model = 32

    def layer_states(self, patches: NDArray[np.float32]) -> NDArray[np.float32]:
        return patches[None].astype(np.float32)


def ecg(**kwargs: object) -> Synthetic:
    return synthetic_ecg(p_mv=P_MV, **kwargs)  # type: ignore[arg-type]


def full_targets(syn: Synthetic, p_waves: bool) -> torch.Tensor:
    """(12, 4992) classes for every sample; nothing masked, since the synthetic truth is known everywhere."""
    classes = labels_for(synthetic_waves(syn, p_waves=p_waves), N_SAMPLES).classes
    return torch.from_numpy(classes.astype(np.int64)).expand(12, -1).clone()


def windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """(recordings, 12, T) -> (recordings * T // window, 12, window)."""
    n, v, t = x.shape
    k = t // window
    return x[:, :, : k * window].reshape(n, v, k, window).permute(0, 2, 1, 3).reshape(n * k, v, window)


@pytest.fixture(scope="module")
def delineators() -> Iterator[dict[str, Delineator]]:
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, os.cpu_count() or 1))  # tiny tensors: more threads only add overhead
    try:
        torch.manual_seed(0)
        recs = [ecg(rr=rr, p_waves=p) for rr, p in TRAIN + VAL]
        signals = torch.from_numpy(np.stack([prepare(s.signal, s.fs) for s in recs]))  # (7, 12, 4992)
        states = torch.from_numpy(np.stack([extract_states(x.numpy(), RawPatchBackbone()) for x in signals]))  # (7, 1, 12, 156, 64)
        targets = torch.stack([full_targets(s, p) for s, (_, p) in zip(recs, TRAIN + VAL)])  # (7, 12, 4992)
        cpu = torch.device("cpu")

        head = ConvHead(n_layers=1, d_in=64, width=32)
        config = TrainConfig(lr=1e-2, batch_size=1, max_epochs=20, patience=20)
        train_jointly([make_trainee("head", head, "states", config)], {"states": TensorSource(states)}, targets, range(6), [6], config, seed=0, device=cpu, log=lambda _: None)

        unet = UNet1D(channels=(8, 16, 32))
        k = N_SAMPLES // WINDOW
        config = TrainConfig(lr=1e-2, batch_size=6, max_epochs=30, patience=30)
        train_jointly([make_trainee("unet", unet, "signal", config)], {"signal": TensorSource(windows(signals, WINDOW))}, windows(targets, WINDOW), range(6 * k), range(6 * k, 7 * k), config, seed=0, device=cpu, log=lambda _: None)

        yield {"head": HeadDelineator("head", head, RawPatchBackbone()), "unet": UNetDelineator("unet", unet)}
    finally:
        torch.set_num_threads(threads)


def beats(points: dict[str, list[float]]) -> Iterator[dict[str, float]]:
    for k in range(len(points["ECG_R_Peaks"])):
        yield {key: points[key][k] for key in KEYS}


@pytest.mark.parametrize("name", ["head", "unet"])
def test_every_lead_gets_the_eleven_keys_with_indices_inside_the_signal(delineators: dict[str, Delineator], name: str) -> None:
    syn = ecg(rr=0.9)

    out = delineators[name].delineate(syn.signal, syn.fs)

    assert set(out) == set(LEADS)
    for lead in LEADS:
        assert tuple(out[lead].points) == KEYS
        assert out[lead].failed is False
        for key in KEYS:
            values = np.array(out[lead].points[key])
            finite = values[np.isfinite(values)]
            assert np.all((finite >= 0) & (finite < N_SAMPLES)), (lead, key)


@pytest.mark.parametrize("name", ["head", "unet"])
def test_waves_come_in_p_qrs_t_order_and_every_beat_is_found(delineators: dict[str, Delineator], name: str) -> None:
    syn = ecg(rr=0.9)

    out = delineators[name].delineate(syn.signal, syn.fs)

    order = ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")
    for lead in LEADS:
        points = out[lead].points
        for beat in beats(points):
            present = [beat[key] for key in order if math.isfinite(beat[key])]
            assert present == sorted(present), (lead, beat)
            assert math.isfinite(beat["ECG_R_Onsets"]) and math.isfinite(beat["ECG_R_Offsets"])
        assert len(points["ECG_R_Peaks"]) == syn.waves["R"].size, lead
        # The true R lies inside a predicted QRS region. (The decoder's R-peak rule follows LUDB and picks the
        # positive deflection, which in the inverted leads aVR and aVL is the S wave, so the peak itself is not compared.)
        on, off = np.array(points["ECG_R_Onsets"]), np.array(points["ECG_R_Offsets"])
        assert np.mean([np.any((on <= r) & (r <= off)) for r in syn.waves["R"]]) >= 0.9, lead
        p = np.array(points["ECG_P_Peaks"])
        assert np.mean(np.isfinite(p)) >= 0.9, lead
        assert np.mean(np.abs(p[np.isfinite(p)] - syn.waves["P"][np.isfinite(p)]) <= 10) >= 0.9, lead  # within 20 ms


@pytest.mark.parametrize("name", ["head", "unet"])
def test_an_af_like_signal_without_p_waves_gets_nan_p_points(delineators: dict[str, Delineator], name: str) -> None:
    syn = ecg(rr=[0.7, 1.0, 0.85, 1.1], p_waves=False)

    out = delineators[name].delineate(syn.signal, syn.fs)

    for lead in LEADS:
        assert len(out[lead].points["ECG_R_Peaks"]) == syn.waves["R"].size, lead
        for key in ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets"):
            assert np.mean(np.isnan(out[lead].points[key])) >= 0.9, (lead, key)


def test_the_head_is_smaller_than_the_unet_and_both_match_the_plan() -> None:
    head, probe, unet = ConvHead(n_layers=20, d_in=2560), LinearProbe(n_layers=20, d_in=2560), UNet1D()

    assert count_parameters(head) == 20 + (2560 * 128 + 128) + 3 * (128 * 128 * 3 + 128) + (128 * 128 + 128)  # 492,180
    assert count_parameters(probe) == 20 + 2560 * 128 + 128
    assert 1_500_000 < count_parameters(unet) < 2_500_000
    assert count_parameters(head) < count_parameters(unet)


def test_models_map_a_recording_to_four_class_scores_per_lead_and_sample() -> None:
    states = torch.zeros(1, 20, 12, 156, 2560, dtype=torch.float16)
    signal = torch.zeros(1, 12, N_SAMPLES)

    with torch.no_grad():
        assert ConvHead(n_layers=20, d_in=2560)(states).shape == (1, 4, 12, N_SAMPLES)
        assert LinearProbe(n_layers=20, d_in=2560)(states).shape == (1, 4, 12, N_SAMPLES)
        assert UNet1D().eval()(signal).shape == (1, 4, 12, N_SAMPLES)
