"""The training loop: masked targets, per-model early stopping with best weights kept, and resume."""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ecgdelin.dataset import LEADS, Wave
from ecgdelin.models import LinearProbe, normalize_layers
from ecgdelin.train import TensorSource, TrainConfig, Trainee, evaluate_loss, lead_targets, make_trainee, predict_classes, targets_for, train_jointly

CPU = torch.device("cpu")
N_BLOCKS, D = 4, 8  # tiny states: (recordings, 1 layer, 12 leads, 4 blocks, 8) -> 128 samples per lead


def tiny_data(seed: int, n: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """States whose first feature decides the class, so a probe can learn them; one fifth of the samples masked."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, 1, 12, N_BLOCKS, D)).astype(np.float16)
    classes = (np.repeat(states[:, 0, :, :, 0], 32, axis=-1) > 0).astype(np.int64)  # (n, 12, 128)
    classes[..., 64:] *= 2  # second half of each lead: class 0 or 2
    targets = torch.from_numpy(classes)
    targets[torch.from_numpy(rng.random(targets.shape) < 0.2)] = -1
    return torch.from_numpy(states), targets


def trainee(name: str, config: TrainConfig, seed: int = 0, source: str = "states") -> Trainee:
    torch.manual_seed(seed)
    return make_trainee(name, LinearProbe(n_layers=1, d_in=D), source, config)


def run(trainees: Sequence[Trainee], states: torch.Tensor, targets: torch.Tensor, config: TrainConfig, checkpoint: Path | None = None) -> None:
    train_jointly(trainees, {"states": TensorSource(states)}, targets, range(4), [4, 5], config, seed=0, device=CPU, checkpoint=checkpoint, log=lambda _: None)


def test_targets_carry_the_class_inside_the_mask_and_minus_one_outside() -> None:
    waves = [Wave("P", 10, 15, 20), Wave("QRS", 30, 35, 40), Wave("T", 50, 60, 70)]

    targets = lead_targets(waves, 100)

    assert targets.shape == (100,) and targets.dtype == torch.int64
    assert torch.all(targets[:10] == -1) and torch.all(targets[71:] == -1)  # outside the marked span
    assert torch.all(targets[10:21] == 1) and torch.all(targets[30:41] == 2) and torch.all(targets[50:71] == 3)
    assert torch.all(targets[21:30] == 0) and torch.all(targets[41:50] == 0)
    assert targets_for({lead: waves for lead in LEADS}, 100).shape == (12, 100)


def test_inputs_at_masked_samples_do_not_affect_training() -> None:
    states, targets = tiny_data(seed=1)
    targets[:, :, 32:64] = -1  # blocks 1 and 3 of every lead are entirely masked ...
    targets[:, :, 96:128] = -1
    altered = states.clone()
    altered[:, :, :, 1] = torch.randn(6, 1, 12, D).half()  # ... so their inputs must not matter
    altered[:, :, :, 3] = torch.randn(6, 1, 12, D).half()
    config = TrainConfig(lr=1e-2, batch_size=2, max_epochs=4, patience=10)

    a, b = trainee("a", config), trainee("b", config)
    run([a], states, targets, config)
    run([b], altered, targets, config)

    assert a.history == b.history
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)


def test_each_model_stops_on_its_own_validation_loss_and_keeps_its_best_weights() -> None:
    states, targets = tiny_data(seed=2)
    flipped = states.clone()
    flipped[4:, ..., 0] *= -1  # the validation recordings contradict the pattern learnt from training
    config = TrainConfig(lr=1e-2, batch_size=2, max_epochs=30, patience=3)
    learns, conflicted = trainee("learns", config), trainee("conflicted", config, seed=1, source="flipped")

    train_jointly(
        [learns, conflicted],
        {"states": TensorSource(states), "flipped": TensorSource(flipped)},
        targets,
        range(4),
        [4, 5],
        config,
        seed=0,
        device=CPU,
        log=lambda _: None,
    )

    assert learns.history[-1]["val_loss"] < learns.history[0]["val_loss"]
    assert conflicted.stopped and len(conflicted.history) < len(learns.history)
    val_losses = [h["val_loss"] for h in conflicted.history]
    assert conflicted.best_epoch == int(np.argmin(val_losses))
    assert len(val_losses) == conflicted.best_epoch + 1 + config.patience
    restored = evaluate_loss(conflicted.model, TensorSource(flipped), targets, [4, 5], config.batch_size, CPU)
    assert math.isclose(restored, min(val_losses), rel_tol=1e-6)


def test_a_run_resumed_from_a_checkpoint_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    states, targets = tiny_data(seed=3)
    straight = TrainConfig(lr=1e-2, batch_size=2, max_epochs=6, patience=10)
    a = trainee("a", straight)
    run([a], states, targets, straight)

    half = TrainConfig(lr=1e-2, batch_size=2, max_epochs=3, patience=10)
    b = trainee("a", half)
    run([b], states, targets, half, checkpoint=tmp_path / "ckpt.pt")
    assert len(b.history) == 3
    resumed = trainee("a", straight, seed=99)  # a fresh, differently initialised model: the checkpoint must overwrite it
    run([resumed], states, targets, straight, checkpoint=tmp_path / "ckpt.pt")

    assert len(resumed.history) == 6
    assert resumed.history == a.history
    for pa, pb in zip(a.model.parameters(), resumed.model.parameters()):
        assert torch.equal(pa, pb)


def test_normalising_the_states_once_per_batch_for_all_models_changes_nothing() -> None:
    """The joint trainer normalises a batch's layers once and tells every head so (``normalized=True``);
    this must equal each head normalising on its own, bit for bit."""
    states, targets = tiny_data(seed=4)
    config = TrainConfig(lr=1e-2, batch_size=2, max_epochs=5, patience=10)
    own = trainee("own", config)
    run([own], states, targets, config)

    torch.manual_seed(0)
    shared = make_trainee("shared", LinearProbe(n_layers=1, d_in=D), "states", config, normalized=True)
    train_jointly([shared], {"states": TensorSource(states, transform=normalize_layers)}, targets, range(4), [4, 5], config, seed=0, device=CPU, log=lambda _: None)

    assert shared.history == own.history
    for pa, pb in zip(own.model.parameters(), shared.model.parameters()):
        assert torch.equal(pa, pb)
    assert np.array_equal(
        predict_classes(own.model, TensorSource(states), [4, 5], 2, CPU),
        predict_classes(shared.model, TensorSource(states, transform=normalize_layers), [4, 5], 2, CPU, {"normalized": True}),
    )
