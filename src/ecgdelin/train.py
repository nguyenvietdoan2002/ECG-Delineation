"""The training loop shared by every trained model (decision 58).

Per-sample cross-entropy ignores samples outside the label mask (target -1). All models of a fold train
in one loop over one data stream: each batch of recordings is moved to the device once per input kind,
and every model takes its own AdamW step on it. Each model stops on its own validation loss and ends
with its best weights. After every epoch the whole state (models, optimisers, shuffling RNG) is
checkpointed, so a preempted container resumes and reproduces the uninterrupted run.
"""

import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from ecgdelin.dataset import LEADS, Wave
from ecgdelin.labels import labels_for

IGNORE = -1


@dataclass(frozen=True)
class TrainConfig:
    """The plan's training table; used as written, no tuning."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 200
    patience: int = 20

    def to_dict(self) -> dict[str, float | int]:
        return {"lr": self.lr, "weight_decay": self.weight_decay, "batch_size": self.batch_size, "max_epochs": self.max_epochs, "patience": self.patience}


def lead_targets(waves: Sequence[Wave], n_samples: int) -> Tensor:
    """One lead's per-sample class, with IGNORE outside the label mask."""
    labels = labels_for(waves, n_samples)
    targets = torch.from_numpy(labels.classes.astype(np.int64))
    targets[~torch.from_numpy(labels.mask)] = IGNORE
    return targets


def targets_for(marks: Mapping[str, Sequence[Wave]], n_samples: int) -> Tensor:
    """(12, n_samples) targets in LEADS order."""
    return torch.stack([lead_targets(marks[lead], n_samples) for lead in LEADS])


class Source(Protocol):
    """Model inputs for a set of recordings, by index."""

    def batch(self, idx: Sequence[int], device: torch.device) -> Tensor: ...

    def prepare(self, batch: Tensor) -> Tensor:
        """Work done once per batch on the device and shared by every model reading this source."""
        ...


class TensorSource:
    """All recordings' inputs in one CPU tensor. On CUDA, batches are gathered into a pinned buffer so
    the copy to the device can overlap with compute. ``transform`` (for example the per-layer
    normalisation of TimesFM states) runs once per batch on the device, for all models at once."""

    def __init__(self, data: Tensor, transform: Callable[[Tensor], Tensor] | None = None) -> None:
        self.data = data
        self.transform = transform
        self._buffers: dict[int, Tensor] = {}

    def prepare(self, batch: Tensor) -> Tensor:
        return batch if self.transform is None else self.transform(batch)

    def batch(self, idx: Sequence[int], device: torch.device) -> Tensor:
        index = torch.as_tensor(list(idx), dtype=torch.long)
        if device.type != "cuda":
            return self.data[index].to(device)
        buffer = self._buffers.get(len(index))
        if buffer is None:
            buffer = torch.empty((len(index), *self.data.shape[1:]), dtype=self.data.dtype).pin_memory()
            self._buffers[len(index)] = buffer
        torch.cuda.current_stream(device).synchronize()  # the previous copy out of this buffer has finished
        torch.index_select(self.data, 0, index, out=buffer)
        return buffer.to(device, non_blocking=True)


@dataclass
class Trainee:
    """One model being trained, with its optimiser and early-stopping state."""

    name: str
    model: nn.Module
    source: str  # which input kind it reads (a key of the sources mapping)
    optimizer: torch.optim.Optimizer
    history: list[dict[str, float]] = field(default_factory=list)  # per epoch: epoch, train_loss, val_loss
    best_val: float = math.inf
    best_epoch: int = -1
    best_state: dict[str, Tensor] | None = None
    bad_epochs: int = 0
    stopped: bool = False
    seconds: float = 0.0  # time spent in this model's forward/backward steps
    kwargs: dict[str, Any] = field(default_factory=dict)  # passed to every model call (e.g. normalized=True)

    def summary(self) -> dict[str, Any]:
        return {
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
            "stopped_early": self.stopped,
            "train_seconds": self.seconds,
            "history": self.history,
        }


def make_trainee(name: str, model: nn.Module, source: str, config: TrainConfig, **kwargs: Any) -> Trainee:
    return Trainee(name, model, source, torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay), kwargs=dict(kwargs))


def _batches(idx: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    for start in range(0, len(idx), batch_size):
        yield list(idx[start : start + batch_size])


def evaluate_loss(
    model: nn.Module, source: Source, targets: Tensor, idx: Sequence[int], batch_size: int, device: torch.device, model_kwargs: Mapping[str, Any] | None = None
) -> float:
    """Mean cross-entropy over all valid samples of the given recordings."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for b in _batches(idx, batch_size):
            y = targets[b].to(device)
            total += F.cross_entropy(model(source.prepare(source.batch(b, device)), **(model_kwargs or {})), y, ignore_index=IGNORE, reduction="sum").item()
            count += int((y != IGNORE).sum().item())
    return total / count if count else math.nan


def predict_classes(
    model: nn.Module, source: Source, idx: Sequence[int], batch_size: int, device: torch.device, model_kwargs: Mapping[str, Any] | None = None
) -> NDArray[np.int8]:
    """(len(idx), 12, samples) argmax classes."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b in _batches(idx, batch_size):
            outs.append(model(source.prepare(source.batch(b, device)), **(model_kwargs or {})).argmax(1).to(torch.int8).cpu())
    return np.asarray(torch.cat(outs).numpy(), dtype=np.int8)


def _save(path: Path, epoch: int, trainees: Sequence[Trainee], rng: np.random.Generator) -> None:
    state = {
        "epoch": epoch,
        "rng": rng.bit_generator.state,
        "trainees": {
            t.name: {
                "model": t.model.state_dict(),
                "optimizer": t.optimizer.state_dict(),
                "history": t.history,
                "best_val": t.best_val,
                "best_epoch": t.best_epoch,
                "best_state": t.best_state,
                "bad_epochs": t.bad_epochs,
                "stopped": t.stopped,
                "seconds": t.seconds,
            }
            for t in trainees
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load(path: Path, trainees: Sequence[Trainee], rng: np.random.Generator) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for t in trainees:
        s = state["trainees"][t.name]
        t.model.load_state_dict(s["model"])
        t.optimizer.load_state_dict(s["optimizer"])
        t.history = s["history"]
        t.best_val, t.best_epoch, t.best_state = s["best_val"], s["best_epoch"], s["best_state"]
        t.bad_epochs, t.stopped, t.seconds = s["bad_epochs"], s["stopped"], s["seconds"]
    rng.bit_generator.state = state["rng"]
    return int(state["epoch"])


def train_jointly(
    trainees: Sequence[Trainee],
    sources: Mapping[str, Source],
    targets: Tensor,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    checkpoint: Path | None = None,
    log: Callable[[str], None] = print,
    after_epoch: Callable[[int], None] | None = None,
) -> None:
    """Train all ``trainees`` over the same shuffled batches of ``train_idx``; ``targets`` is (recordings, 12, samples).
    ``after_epoch`` runs once the epoch's checkpoint is written (the Modal script commits the Volume there)."""
    train_idx, val_idx = list(train_idx), list(val_idx)
    rng = np.random.default_rng(seed)
    for t in trainees:
        t.model.to(device)
    start = 0
    if checkpoint is not None and checkpoint.exists():
        start = _load(checkpoint, trainees, rng)
        log(f"resumed from {checkpoint} after {start} epochs")

    for epoch in range(start, config.max_epochs):
        active = [t for t in trainees if not t.stopped]
        if not active:
            break
        order = rng.permutation(len(train_idx))
        sums = {t.name: 0.0 for t in active}
        n_batches = 0
        for t in active:
            t.model.train()
        for b in _batches([train_idx[i] for i in order], config.batch_size):
            y = targets[b].to(device)
            inputs: dict[str, Tensor] = {}
            for t in active:
                tick = time.perf_counter()
                if t.source not in inputs:
                    inputs[t.source] = sources[t.source].prepare(sources[t.source].batch(b, device))
                loss = F.cross_entropy(t.model(inputs[t.source], **t.kwargs), y, ignore_index=IGNORE)
                t.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                t.optimizer.step()
                sums[t.name] += loss.item()
                t.seconds += time.perf_counter() - tick
            n_batches += 1

        for t in active:
            tick = time.perf_counter()
            val = evaluate_loss(t.model, sources[t.source], targets, val_idx, config.batch_size, device, t.kwargs)
            t.seconds += time.perf_counter() - tick
            t.history.append({"epoch": epoch, "train_loss": sums[t.name] / n_batches, "val_loss": val})
            if val < t.best_val:
                t.best_val, t.best_epoch, t.bad_epochs = val, epoch, 0
                t.best_state = {k: v.detach().clone().cpu() for k, v in t.model.state_dict().items()}
            else:
                t.bad_epochs += 1
                if t.bad_epochs >= config.patience:
                    t.stopped = True
            log(f"epoch {epoch:3d} {t.name:>8s}: train {sums[t.name] / n_batches:.4f} val {val:.4f} best {t.best_val:.4f}@{t.best_epoch}{' stopped' if t.stopped else ''}")
        if checkpoint is not None:
            _save(checkpoint, epoch + 1, trainees, rng)
        if after_epoch is not None:
            after_epoch(epoch)

    for t in trainees:
        if t.best_state is not None:
            t.model.load_state_dict(t.best_state)
