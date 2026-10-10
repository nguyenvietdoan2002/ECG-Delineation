"""Frozen TimesFM 3 hidden states for every lead and block, forward and time-reversed (decisions 10, 30).

A recording's 12 leads are the 12 variates of one TimesFM input: 156 blocks of 32 samples each.
Every transformer layer's output is captured (20 x 1280 per lead and block). Because TimesFM's
attention over blocks is causal, a second pass runs on the time-reversed signal and is flipped back,
so each block also gets a view of what follows it. The two passes are joined to 2560 values per layer.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

N_SAMPLES = 4992
PATCH = 32
N_BLOCKS = N_SAMPLES // PATCH  # 156
HF_REPO = "google/timesfm-3.0-pytorch"


class Backbone(Protocol):
    n_layers: int
    d_model: int

    def layer_states(self, patches: NDArray[np.float32]) -> NDArray[np.float32]:
        """(variates, blocks, PATCH) -> (n_layers, variates, blocks, d_model): each layer's output."""
        ...


def to_patches(signal: NDArray[np.floating]) -> NDArray[np.float32]:
    """(leads, 4992) -> (leads, 156, 32)."""
    leads, n = signal.shape
    if n != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} samples per lead, got {n}")
    return np.ascontiguousarray(signal, dtype=np.float32).reshape(leads, N_BLOCKS, PATCH)


def pass_states(signal: NDArray[np.floating], backbone: Backbone, reverse: bool) -> NDArray[np.float32]:
    """One backbone pass. The reversed pass runs on the time-reversed signal and its block axis is
    flipped back, so block k of either result covers the same 32 samples."""
    x = signal[:, ::-1] if reverse else signal
    states = backbone.layer_states(to_patches(x))
    return np.ascontiguousarray(states[:, :, ::-1, :] if reverse else states, dtype=np.float32)


def extract_states(signal: NDArray[np.floating], backbone: Backbone) -> NDArray[np.float16]:
    """(leads, 4992) -> (n_layers, leads, 156, 2 * d_model) in fp16: [forward | reversed] per block."""
    forward = pass_states(signal, backbone, reverse=False)
    reverse = pass_states(signal, backbone, reverse=True)
    return np.concatenate([forward, reverse], axis=-1).astype(np.float16)


class TimesFMBackbone:
    """The real model from the Hugging Face checkpoint, run in fp32 under inference mode."""

    def __init__(self, revision: str | None = None, device: str | None = None) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from timesfm3.torch import TimesFM3Torch

        config_path = hf_hub_download(HF_REPO, "config.json", revision=revision)
        self.revision = Path(config_path).parent.name  # the snapshot folder is the commit hash
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimesFM3Torch.from_pretrained(HF_REPO, revision=self.revision).to(self.device).eval()
        self.layers: Sequence[Any] = self.model.transformer_stack.layers
        self.n_layers = len(self.layers)
        self.d_model = int(self.model.transformer_config.transformer.model_dims)

    def layer_states(self, patches: NDArray[np.float32]) -> NDArray[np.float32]:
        import torch

        v, n, p = patches.shape
        values = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))[None].to(self.device)
        inputs = {
            "values": values,
            "masks": torch.zeros_like(values, dtype=torch.bool),
            # All leads are forecast targets, as in TimesFM's own decode: the model sees each block's
            # values but not the "future covariate" copy of what follows, so states stay causal.
            "patch_is_target": torch.ones((1, v, n), dtype=torch.bool, device=self.device),
        }
        captured: list[Any] = []
        hooks = [layer.register_forward_hook(lambda _m, _i, out: captured.append(out[0].detach())) for layer in self.layers]
        try:
            with torch.inference_mode():
                self.model(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        return np.asarray(torch.stack(captured)[:, 0].float().cpu().numpy(), dtype=np.float32)
