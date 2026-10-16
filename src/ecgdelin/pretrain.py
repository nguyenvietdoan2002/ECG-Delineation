"""Pretraining one head on PTB-XL states streamed from files (decisions 76 and 77).

Step 1's trainer holds every state in memory; the 5,000 PTB-XL ECGs (about 0.96 TB) do not fit, so this
loop reads each batch's files from the Volume while the previous batch trains. The loss, the model call
and the per-layer normalisation are step 1's. The stopping set's loss is checked ``checks_per_epoch``
times per epoch; training stops after ``patience_epochs`` epochs without improvement or after
``max_epochs``, and ends with the best weights. A checkpoint after every check lets a preempted
container resume where it stopped; each epoch's order depends only on the seed and the epoch.
"""

import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from ecgdelin.preprocess import N_SAMPLES
from ecgdelin.train import IGNORE, evaluate_loss


@dataclass(frozen=True)
class PretrainConfig:
    lr: float = 1e-3  # step 1's TrainConfig
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 20
    checks_per_epoch: int = 2
    patience_epochs: int = 3

    def to_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def targets_from_labels(classes: NDArray[np.integer], mask: NDArray[np.bool_]) -> NDArray[np.int8]:
    """(12, samples) classes and mask -> (12, N_SAMPLES) int8 targets, IGNORE where masked (step 1's crop)."""
    out = classes[:, :N_SAMPLES].astype(np.int8)
    out[~mask[:, :N_SAMPLES]] = IGNORE
    return out


class Targets:
    """(recordings, 12, N_SAMPLES) int8 targets; indexing returns int64, as cross-entropy needs."""

    def __init__(self, data: NDArray[np.int8]) -> None:
        self.data = torch.from_numpy(np.ascontiguousarray(data))

    def __getitem__(self, idx: Sequence[int]) -> Tensor:
        return self.data[torch.as_tensor(list(idx), dtype=torch.long)].long()

    def __len__(self) -> int:
        return int(self.data.shape[0])


class FileSource:
    """States read from one ``.npy`` per recording, ``threads`` files at a time. ``transform`` (step 1's
    ``normalize_layers``) runs on the device. ``load`` returns a CPU tensor so the next batch can be read
    while the current one trains; ``batch`` does both steps, for ``evaluate_loss``."""

    def __init__(self, paths: Sequence[Path], transform: Callable[[Tensor], Tensor] | None = None, threads: int = 16) -> None:
        self.paths = list(paths)
        self.transform = transform
        self.pool = ThreadPoolExecutor(threads)

    def load(self, idx: Sequence[int]) -> Tensor:
        arrays = list(self.pool.map(lambda i: np.load(self.paths[i]), idx))
        return torch.from_numpy(np.stack(arrays))

    def prepare(self, batch: Tensor) -> Tensor:
        return batch if self.transform is None else self.transform(batch)

    def batch(self, idx: Sequence[int], device: torch.device) -> Tensor:
        return self.load(idx).to(device, non_blocking=True)


class MemorySource:
    """States already in memory, one CPU tensor (recordings, layers, leads, blocks, d) (decision 80). Same
    interface as ``FileSource``: ``load`` gathers a batch on the CPU, ``batch`` also moves it to the device."""

    def __init__(self, data: Tensor, transform: Callable[[Tensor], Tensor] | None = None) -> None:
        self.data = data
        self.transform = transform

    def load(self, idx: Sequence[int]) -> Tensor:
        return torch.index_select(self.data, 0, torch.as_tensor(list(idx), dtype=torch.long))

    def prepare(self, batch: Tensor) -> Tensor:
        return batch if self.transform is None else self.transform(batch)

    def batch(self, idx: Sequence[int], device: torch.device) -> Tensor:
        return self.load(idx).to(device, non_blocking=True)


def assign_leads(n: int, per_ecg: int, seed: int, n_leads: int = 12) -> list[list[int]]:
    """``per_ecg`` distinct leads for each of ``n`` recordings, every lead used equally often: each block
    of ``n_leads // per_ecg`` recordings shares one random permutation of the leads (decision 80)."""
    if n_leads % per_ecg:
        raise ValueError(f"{per_ecg} does not divide {n_leads}")
    rng = np.random.default_rng(seed)
    group = n_leads // per_ecg
    out: list[list[int]] = []
    while len(out) < n:
        perm = rng.permutation(n_leads)
        out.extend(sorted(int(v) for v in perm[g * per_ecg : (g + 1) * per_ecg]) for g in range(group))
    return out[:n]


def read_npy(path: Path) -> NDArray[Any]:
    """``np.load`` with a plain file read: Python's ``readinto`` releases the interpreter lock while it
    waits on the Volume, which numpy's reader did not, starving the container's heartbeat (decision 80)."""
    with open(path, "rb") as f:
        major, _minor = np.lib.format.read_magic(f)
        header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = header(f)
        if fortran:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        out = np.empty(shape, dtype=dtype)
        view = memoryview(out.reshape(-1).view(np.uint8))
        done = 0
        while done < len(view):
            n = f.readinto(view[done:])
            if not n:
                raise EOFError(f"{path}: {done} of {len(view)} bytes")
            done += n
    return out


def _chunks(order: Sequence[int], batch_size: int) -> list[list[int]]:
    return [list(order[k : k + batch_size]) for k in range(0, len(order), batch_size)]


def _prefetched(source: FileSource, batches: Sequence[list[int]], depth: int = 3) -> Iterator[tuple[list[int], Tensor]]:
    """Each batch's CPU tensor, with the next ``depth`` batches read in the background."""
    with ThreadPoolExecutor(depth) as ahead:
        pending = [ahead.submit(source.load, b) for b in batches[:depth]]
        for k, b in enumerate(batches):
            data = pending.pop(0).result()
            if k + depth < len(batches):
                pending.append(ahead.submit(source.load, batches[k + depth]))
            yield b, data


def pretrain(
    model: nn.Module,
    source: FileSource,
    targets: Targets,
    train_idx: Sequence[int],
    stop_idx: Sequence[int],
    config: PretrainConfig,
    seed: int,
    device: torch.device,
    checkpoint: Path | None = None,
    log: Callable[[str], None] = print,
    after_check: Callable[[int], None] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train ``model`` in place; returns the history and the best check. ``model`` ends with its best weights."""
    kwargs = model_kwargs or {}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    patience = config.patience_epochs * config.checks_per_epoch
    state: dict[str, Any] = {"check": 0, "best_val": math.inf, "best_check": -1, "bad": 0, "history": [], "seconds": 0.0, "stopped": False}
    best: dict[str, Tensor] | None = None
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state, best = saved["state"], saved["best"]
        log(f"resumed from {checkpoint} after {state['check']} checks")

    total = config.max_epochs * config.checks_per_epoch
    while state["check"] < total and not state["stopped"]:
        epoch, part = divmod(state["check"], config.checks_per_epoch)
        order = np.random.default_rng([seed, epoch]).permutation(len(train_idx))
        batches = _chunks([train_idx[i] for i in order], config.batch_size)
        bounds = np.linspace(0, len(batches), config.checks_per_epoch + 1).round().astype(int)
        tick = time.perf_counter()
        model.train()
        losses = []
        for b, data in _prefetched(source, batches[bounds[part] : bounds[part + 1]]):
            x = source.prepare(data.to(device, non_blocking=True))
            loss = F.cross_entropy(model(x, **kwargs), targets[b].to(device), ignore_index=IGNORE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val = evaluate_loss(model, source, targets, stop_idx, config.batch_size, device, kwargs)
        state["seconds"] += time.perf_counter() - tick
        state["history"].append({"check": state["check"], "epoch": epoch + (part + 1) / config.checks_per_epoch,
                                 "train_loss": float(np.mean(losses)) if losses else math.nan, "val_loss": val})
        if val < state["best_val"]:
            state.update(best_val=val, best_check=state["check"], bad=0)
            best = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
            state["stopped"] = state["bad"] >= patience
        log(f"check {state['check']:2d} (epoch {epoch + (part + 1) / config.checks_per_epoch:4.1f}): train {state['history'][-1]['train_loss']:.4f} "
            f"stop-set {val:.4f} best {state['best_val']:.4f}@{state['best_check']}{' stopped' if state['stopped'] else ''} ({time.perf_counter() - tick:.0f} s)")
        state["check"] += 1
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state, "best": best}, tmp)
            os.replace(tmp, checkpoint)
        if after_check is not None:
            after_check(state["check"])
    if best is not None:
        model.load_state_dict(best)
    return {**state, "best_epoch": (state["best_check"] + 1) / config.checks_per_epoch}
