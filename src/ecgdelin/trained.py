"""Trained models behind the Delineator interface (spec story 26): a 12-lead ECG in, points per lead out.

The head wrapper runs the backbone for the states, the U-Net wrapper runs the network on the prepared
signal; both take the argmax class per sample and hand the regions to the decoder together with the
high-passed signal the model saw (decision 58).
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ecgdelin.dataset import LEADS
from ecgdelin.decode import regions_to_points
from ecgdelin.delineate import LeadPoints
from ecgdelin.preprocess import prepare
from ecgdelin.states import Backbone, extract_states


def points_from_classes(classes: NDArray[np.int8], signal: NDArray[np.floating], fs: int) -> dict[str, LeadPoints]:
    """(12, samples) classes and the matching signal -> points per lead."""
    return {lead: regions_to_points(classes[k], np.asarray(signal[k], dtype=np.float64), fs) for k, lead in enumerate(LEADS)}


def classes_of(model: nn.Module, inputs: Tensor, **kwargs: Any) -> NDArray[np.int8]:
    """One recording's (12, samples) argmax classes from a model's scores for ``inputs`` of batch size 1."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.asarray(model(inputs.to(device), **kwargs).argmax(1)[0].to(torch.int8).cpu().numpy(), dtype=np.int8)


class HeadDelineator:
    """A TimesFM-state head (probe or conv head) with the backbone that produces its states."""

    def __init__(self, name: str, model: nn.Module, backbone: Backbone) -> None:
        self.name = name
        self.model = model
        self.backbone = backbone

    def delineate(self, signal: NDArray[np.float64], fs: int) -> dict[str, LeadPoints]:
        x = prepare(signal, fs)
        states = torch.from_numpy(extract_states(x, self.backbone))[None]  # (1, layers, 12, 156, 2d) fp16
        return points_from_classes(classes_of(self.model, states), x, fs)


class UNetDelineator:
    def __init__(self, name: str, model: nn.Module) -> None:
        self.name = name
        self.model = model

    def delineate(self, signal: NDArray[np.float64], fs: int) -> dict[str, LeadPoints]:
        x = prepare(signal, fs)
        return points_from_classes(classes_of(self.model, torch.from_numpy(x)[None]), x, fs)
