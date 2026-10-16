import numpy as np
import torch
from torch import nn

from ecgdelin.preprocess import N_SAMPLES
from ecgdelin.pretrain import (
    FileSource,
    MemorySource,
    PretrainConfig,
    Targets,
    assign_leads,
    pretrain,
    read_npy,
    targets_from_labels,
)
from ecgdelin.train import IGNORE


class Tiny(nn.Module):
    """(b, 2, 12, 3) 'states' -> (b, 4 classes, 12, N_SAMPLES) scores, enough to exercise the loop."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.lin(x.float().mean(1))  # (b, 12, 4)
        return s.permute(0, 2, 1)[..., None].expand(-1, -1, -1, N_SAMPLES)


def _data(tmp_path, n: int = 12):
    rng = np.random.default_rng(0)
    paths, targets = [], []
    for k in range(n):
        label = k % 4
        x = rng.normal(size=(2, 12, 3)).astype(np.float16) + np.float16(label)
        p = tmp_path / f"{k}.npy"
        np.save(p, x)
        paths.append(p)
        t = np.full((12, N_SAMPLES), label, dtype=np.int8)
        t[:, :100] = IGNORE
        targets.append(t)
    return paths, Targets(np.stack(targets))


def test_targets_crop_and_ignore_masked_samples() -> None:
    classes = np.ones((12, 5000), dtype=np.int8)
    mask = np.ones((12, 5000), dtype=bool)
    mask[3, :10] = False
    t = targets_from_labels(classes, mask)
    assert t.shape == (12, N_SAMPLES) and t.dtype == np.int8
    assert (t[3, :10] == IGNORE).all() and (t[3, 10:] == 1).all()
    assert Targets(t[None])[[0]].dtype == torch.int64


def test_file_source_reads_files_in_order(tmp_path) -> None:
    paths, _ = _data(tmp_path, 4)
    batch = FileSource(paths).load([2, 0])
    assert batch.shape == (2, 2, 12, 3)
    assert torch.equal(batch[0], torch.from_numpy(np.load(paths[2])))


def test_read_npy_matches_numpy(tmp_path) -> None:
    a = np.random.default_rng(1).normal(size=(3, 4, 5)).astype(np.float16)
    np.save(tmp_path / "a.npy", a)
    b = read_npy(tmp_path / "a.npy")
    assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)


def test_assign_leads_is_balanced_distinct_and_seeded() -> None:
    leads = assign_leads(4500, 4, seed=0)
    assert len(leads) == 4500 and all(len(set(x)) == 4 for x in leads)
    counts = np.bincount(np.concatenate(leads), minlength=12)
    assert (counts == 1500).all()
    assert leads == assign_leads(4500, 4, seed=0) and leads != assign_leads(4500, 4, seed=1)
    stop = assign_leads(500, 4, seed=0)
    assert len(stop) == 500 and np.bincount(np.concatenate(stop), minlength=12).min() >= 166


def test_memory_source_matches_file_source(tmp_path) -> None:
    paths, targets = _data(tmp_path, 6)
    data = torch.from_numpy(np.stack([np.load(p) for p in paths]))
    mem, files = MemorySource(data), FileSource(paths)
    assert torch.equal(mem.load([4, 1]), files.load([4, 1]))
    torch.manual_seed(0)
    a = pretrain(Tiny(), mem, targets, [0, 1, 2, 3], [4, 5], PretrainConfig(lr=0.05, batch_size=2, max_epochs=1), 0, torch.device("cpu"))
    torch.manual_seed(0)
    b = pretrain(Tiny(), files, targets, [0, 1, 2, 3], [4, 5], PretrainConfig(lr=0.05, batch_size=2, max_epochs=1), 0, torch.device("cpu"))
    assert [h["val_loss"] for h in a["history"]] == [h["val_loss"] for h in b["history"]]


def test_pretrain_learns_stops_and_keeps_the_best(tmp_path) -> None:
    paths, targets = _data(tmp_path)
    torch.manual_seed(0)
    model = Tiny()
    config = PretrainConfig(lr=0.05, batch_size=4, max_epochs=6, checks_per_epoch=2, patience_epochs=1)
    out = pretrain(model, FileSource(paths), targets, list(range(8)), [8, 9, 10, 11], config, seed=0, device=torch.device("cpu"))
    vals = [h["val_loss"] for h in out["history"]]
    assert len(vals) <= 12 and vals[out["best_check"]] == min(vals)
    assert min(vals) < vals[0]


def test_pretrain_resumes_from_its_checkpoint(tmp_path) -> None:
    paths, targets = _data(tmp_path)
    config = PretrainConfig(lr=0.05, batch_size=4, max_epochs=2, checks_per_epoch=2, patience_epochs=5)
    torch.manual_seed(0)
    whole = pretrain(Tiny(), FileSource(paths), targets, list(range(8)), [8, 9, 10, 11], config, 0, torch.device("cpu"))
    ckpt = tmp_path / "ckpt.pt"
    torch.manual_seed(0)
    first = PretrainConfig(lr=0.05, batch_size=4, max_epochs=1, checks_per_epoch=2, patience_epochs=5)
    pretrain(Tiny(), FileSource(paths), targets, list(range(8)), [8, 9, 10, 11], first, 0, torch.device("cpu"), checkpoint=ckpt)
    resumed = pretrain(Tiny(), FileSource(paths), targets, list(range(8)), [8, 9, 10, 11], config, 0, torch.device("cpu"), checkpoint=ckpt)
    assert [h["val_loss"] for h in resumed["history"]] == [h["val_loss"] for h in whole["history"]]
