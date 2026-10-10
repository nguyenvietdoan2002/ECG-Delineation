"""The real TimesFM 3 backbone on one LUDB record: shapes, fp16 safety, causality, reversed context,
and lead interaction through variate attention. Needs the weights and a GPU: runs on Modal
(`modal run scripts/gpu_tests.py`)."""

from pathlib import Path

import numpy as np
import pytest

from ecgdelin.dataset import load_record
from ecgdelin.preprocess import prepare
from ecgdelin.states import N_BLOCKS, TimesFMBackbone, extract_states, pass_states

pytestmark = pytest.mark.gpu
CHANGE_FROM = 4000  # samples; block 125 onwards
BLOCK = CHANGE_FROM // 32


@pytest.fixture(scope="module")
def backbone() -> TimesFMBackbone:
    return TimesFMBackbone()


@pytest.fixture(scope="module")
def signal(ludb_root: Path) -> np.ndarray:
    rec = load_record(ludb_root, "1")
    return prepare(rec.signal, rec.fs)


def disturbed(signal: np.ndarray, leads: slice = slice(None)) -> np.ndarray:
    out = signal.copy()
    out[leads, CHANGE_FROM:] += np.random.default_rng(0).normal(0.0, 0.5, out[leads, CHANGE_FROM:].shape).astype(np.float32)
    return out


def test_states_have_the_planned_shape_and_fit_in_fp16(backbone: TimesFMBackbone, signal: np.ndarray) -> None:
    assert (backbone.n_layers, backbone.d_model) == (20, 1280)

    states = extract_states(signal, backbone)

    assert states.shape == (20, 12, N_BLOCKS, 2560)
    assert states.dtype == np.float16
    assert np.all(np.isfinite(states))
    assert np.max(np.abs(states.astype(np.float32))) < 10_000  # fp16 limit is 65,504

    forward = pass_states(signal, backbone, reverse=False)
    error = np.abs(states[..., :1280].astype(np.float32) - forward)
    assert np.max(error) / np.max(np.abs(forward)) < 1e-3


def test_forward_states_are_causal_in_time(backbone: TimesFMBackbone, signal: np.ndarray) -> None:
    before = pass_states(signal, backbone, reverse=False)
    after = pass_states(disturbed(signal), backbone, reverse=False)

    np.testing.assert_allclose(after[:, :, :BLOCK], before[:, :, :BLOCK], atol=1e-3, rtol=1e-3)
    assert np.max(np.abs(after[:, :, BLOCK:] - before[:, :, BLOCK:])) > 1e-2


def test_reversed_states_carry_what_follows_a_block(backbone: TimesFMBackbone, signal: np.ndarray) -> None:
    before = pass_states(signal, backbone, reverse=True)
    after = pass_states(disturbed(signal), backbone, reverse=True)

    assert np.max(np.abs(after[:, :, :BLOCK] - before[:, :, :BLOCK])) > 1e-2  # earlier blocks see the change
    assert np.max(np.abs(after[:, :, 0] - before[:, :, 0])) > 1e-3  # even the first block


def test_leads_interact_through_variate_attention(backbone: TimesFMBackbone, signal: np.ndarray) -> None:
    before = pass_states(signal, backbone, reverse=False)
    after = pass_states(disturbed(signal, leads=slice(11, 12)), backbone, reverse=False)  # only V6 changes

    assert np.max(np.abs(after[:, 0, BLOCK:] - before[:, 0, BLOCK:])) > 1e-3  # lead I's states change
    np.testing.assert_allclose(after[:, 0, :BLOCK], before[:, 0, :BLOCK], atol=1e-3, rtol=1e-3)  # but still causally
