import numpy as np
import torch
from torch import nn

from ecgdelin.preprocess import N_SAMPLES
from ecgdelin.pretrain import FileSource, PretrainConfig, Targets, pretrain, targets_from_labels
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
