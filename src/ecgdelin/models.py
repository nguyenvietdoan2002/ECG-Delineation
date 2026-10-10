"""The trained models: the TimesFM-state heads (linear probe, conv head) and the 1-D U-Net control.

Every model maps a batch to class scores of shape (batch, 4 classes, 12 leads, samples), so one loss,
one decoder and one evaluation path serve all of them (decisions 27, 55, 56, 57).
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ecgdelin.labels import CLASSES

N_CLASSES = len(CLASSES)
PATCH = 32


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def normalize_layers(states: Tensor) -> Tensor:
    """(b, L, v, n, d) -> fp16 copy with every layer's d-vector at zero mean and unit variance (decision 56)."""
    out = torch.empty(states.shape, dtype=torch.float16, device=states.device)
    for layer in range(states.shape[1]):
        out[:, layer] = F.layer_norm(states[:, layer].float(), (states.shape[-1],)).half()
    return out


class _Blend(torch.autograd.Function):
    """sum over layers of w_l * x_l in fp32, saving only the fp16 input for the backward pass."""

    @staticmethod
    def forward(ctx: Any, weights: Tensor, states: Tensor) -> Tensor:
        ctx.save_for_backward(states)
        out = torch.zeros((states.shape[0], *states.shape[2:]), dtype=torch.float32, device=states.device)
        for layer in range(states.shape[1]):
            out.add_(states[:, layer].float(), alpha=float(weights[layer]))
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None]:
        (states,) = ctx.saved_tensors
        grad_w = torch.stack([(grad_out * states[:, layer].float()).sum() for layer in range(states.shape[1])])
        return grad_w, None


class LayerBlend(nn.Module):
    """Softmax-weighted sum over the layers of per-layer-normalised states."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, states: Tensor) -> Tensor:
        """(b, L, v, n, d) raw states (fp16 or fp32) -> (b, v, n, d) fp32."""
        out: Tensor = _Blend.apply(torch.softmax(self.logits, 0), normalize_layers(states))
        return out

    def weights(self) -> list[float]:
        return [float(w) for w in torch.softmax(self.logits.detach(), 0)]


def _to_scores(x: Tensor) -> Tensor:
    """(b, v, n, PATCH * classes) -> (b, classes, v, n * PATCH): each block's 32 samples laid end to end."""
    b, v, n, _ = x.shape
    return x.reshape(b, v, n * PATCH, N_CLASSES).permute(0, 3, 1, 2)


class LinearProbe(nn.Module):
    """Decision 27 (a): the layer blend, then one linear layer d_in -> 32 samples x 4 classes."""

    def __init__(self, n_layers: int, d_in: int) -> None:
        super().__init__()
        self.blend = LayerBlend(n_layers)
        self.out = nn.Linear(d_in, PATCH * N_CLASSES)

    def forward(self, states: Tensor) -> Tensor:
        return _to_scores(self.out(self.blend(states)))


class ConvHead(nn.Module):
    """Decision 27 (b): blend -> linear d_in -> width -> three width-3 convolutions along the blocks of each
    lead (receptive field +-3 blocks, about +-200 ms) -> linear width -> 32 samples x 4 classes."""

    def __init__(self, n_layers: int, d_in: int, width: int = 128) -> None:
        super().__init__()
        self.blend = LayerBlend(n_layers)
        self.inp = nn.Linear(d_in, width)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(3)])
        self.out = nn.Linear(width, PATCH * N_CLASSES)

    def forward(self, states: Tensor) -> Tensor:
        x = F.relu(self.inp(self.blend(states)))  # (b, v, n, width)
        b, v, n, w = x.shape
        x = x.reshape(b * v, n, w).transpose(1, 2)  # (b * v, width, n)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.transpose(1, 2).reshape(b, v, n, w)
        return _to_scores(self.out(x))


class _ConvBlock(nn.Sequential):
    """Two Conv1d-BatchNorm-ReLU steps."""

    def __init__(self, c_in: int, c_out: int, kernel: int) -> None:
        pad = kernel // 2
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet1D(nn.Module):
    """Decision 57: a standard 1-D U-Net. Input (b, 12, T) in mV, standardised per lead inside (decision 55);
    output (b, 4 classes, 12 leads, T). T must be divisible by 2 ** (levels - 1)."""

    def __init__(self, n_leads: int = 12, channels: Sequence[int] = (16, 32, 64, 128, 256), kernel: int = 9) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.factor = 2 ** (len(channels) - 1)
        widths = [n_leads, *channels]
        self.down = nn.ModuleList([_ConvBlock(widths[k], widths[k + 1], kernel) for k in range(len(channels))])
        self.up = nn.ModuleList([nn.ConvTranspose1d(channels[k], channels[k - 1], 2, stride=2) for k in range(len(channels) - 1, 0, -1)])
        self.dec = nn.ModuleList([_ConvBlock(2 * channels[k - 1], channels[k - 1], kernel) for k in range(len(channels) - 1, 0, -1)])
        self.out = nn.Conv1d(channels[0], n_leads * N_CLASSES, 1)

    def forward(self, signal: Tensor) -> Tensor:
        b, v, t = signal.shape
        if t % self.factor:
            raise ValueError(f"signal length {t} is not divisible by {self.factor}")
        x = (signal - signal.mean(-1, keepdim=True)) / (signal.std(-1, keepdim=True) + 1e-6)
        skips = []
        for k, block in enumerate(self.down):
            x = block(x)
            if k < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool1d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.out(x).reshape(b, v, N_CLASSES, t).permute(0, 2, 1, 3)
