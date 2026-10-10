"""The state extractor: forward and time-reversed backbone passes joined per block, with a stub backbone."""

import numpy as np
from numpy.typing import NDArray

from ecgdelin.states import N_BLOCKS, N_SAMPLES, PATCH, extract_states, pass_states, to_patches


class FirstSampleBackbone:
    """Layer l's state for a patch is (first sample of the patch + l), repeated over d_model."""

    n_layers = 3
    d_model = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def layer_states(self, patches: NDArray[np.float32]) -> NDArray[np.float32]:
        self.calls.append(patches.shape)
        first = patches[..., 0]  # (v, n)
        return np.stack([np.repeat((first + layer)[..., None], self.d_model, axis=-1) for layer in range(self.n_layers)]).astype(np.float32)


def ramp() -> NDArray[np.float32]:
    return np.tile(np.arange(N_SAMPLES, dtype=np.float32), (12, 1))


def test_to_patches_splits_each_lead_into_156_blocks_of_32() -> None:
    patches = to_patches(ramp())

    assert patches.shape == (12, N_BLOCKS, PATCH)
    assert patches[3, 5, 0] == 5 * 32
    assert patches[3, 5, -1] == 5 * 32 + 31


def test_forward_pass_states_follow_the_blocks_in_time_order() -> None:
    backbone = FirstSampleBackbone()

    states = pass_states(ramp(), backbone, reverse=False)

    assert states.shape == (3, 12, N_BLOCKS, 2)
    assert backbone.calls == [(12, N_BLOCKS, PATCH)]
    for layer in range(3):
        np.testing.assert_array_equal(states[layer, 0, :, 0], 32 * np.arange(N_BLOCKS) + layer)


def test_reversed_pass_sees_time_backwards_and_is_flipped_back_to_the_forward_block_order() -> None:
    states = pass_states(ramp(), FirstSampleBackbone(), reverse=True)

    # The backbone's "first sample" of reversed block j is the last sample of original block 155 - j;
    # after flipping, block k holds the last sample of original block k.
    np.testing.assert_array_equal(states[0, 0, :, 0], 32 * np.arange(N_BLOCKS) + 31)


def test_extract_states_joins_forward_and_reversed_halves_in_fp16() -> None:
    states = extract_states(ramp(), FirstSampleBackbone())

    assert states.shape == (3, 12, N_BLOCKS, 4)
    assert states.dtype == np.float16
    np.testing.assert_array_equal(states[1, 7, 10, :2], [10 * 32 + 1, 10 * 32 + 1])  # forward half
    np.testing.assert_array_equal(states[1, 7, 10, 2:], [10 * 32 + 31 + 1, 10 * 32 + 31 + 1])  # reversed half
