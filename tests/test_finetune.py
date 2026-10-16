"""Arm B's fine-tuning: heads start from the given weights, per-seed streams differ, resume, adoption rule."""

from pathlib import Path

import torch

from ecgdelin.finetune import adoption, finetune_heads
from ecgdelin.models import LinearProbe
from ecgdelin.score import Verdict
from ecgdelin.train import TensorSource, TrainConfig, make_trainee, train_jointly
from test_train import CPU, D, tiny_data

CONFIG = TrainConfig(lr=1e-2, batch_size=2, max_epochs=6, patience=10)


def probe() -> LinearProbe:
    return LinearProbe(n_layers=1, d_in=D)


def pretrained(seed: int = 7) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in probe().state_dict().items()}


def heads(init, streams: str, config: TrainConfig = CONFIG, checkpoint_dir: Path | None = None):
    states, targets = tiny_data(seed=5, n=8)
    return finetune_heads(probe, init, {"states": TensorSource(states)}, targets, range(6), [6, 7], config, CPU,
                          streams=streams, checkpoint_dir=checkpoint_dir, run="t", log=lambda _: None)


def same(a, b) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.model.parameters(), b.model.parameters()))


def test_heads_start_from_the_given_weights() -> None:
    init = pretrained()
    (t,) = heads({0: init}, "per_seed", TrainConfig(max_epochs=0))
    assert t.name == "head_s0"
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, init[k])


def test_one_start_with_one_stream_repeats_one_run_so_seeds_need_their_own_streams() -> None:
    init = pretrained()
    shared = heads({0: init, 1: init, 2: init}, "shared")
    assert same(shared[0], shared[1]) and same(shared[0], shared[2])
    per_seed = heads({0: init, 1: init, 2: init}, "per_seed")
    assert not same(per_seed[0], per_seed[1]) and not same(per_seed[1], per_seed[2])
    assert same(per_seed[0], shared[0])  # seed 0's stream is step 1's stream


def test_seed_zero_matches_step_one_trainer_from_the_same_start() -> None:
    init = pretrained()
    (ours,) = heads({0: init}, "per_seed")
    states, targets = tiny_data(seed=5, n=8)
    model = probe()
    model.load_state_dict(init)
    ref = make_trainee("ref", model, "states", CONFIG, normalized=True)
    train_jointly([ref], {"states": TensorSource(states)}, targets, range(6), [6, 7], CONFIG, 0, CPU, log=lambda _: None)
    assert ours.history == ref.history and same(ours, ref)


def test_a_preempted_run_resumes_from_its_per_seed_checkpoints(tmp_path: Path) -> None:
    init = {0: pretrained(1), 1: pretrained(2)}
    straight = heads(init, "per_seed")
    half = TrainConfig(lr=1e-2, batch_size=2, max_epochs=3, patience=10)
    heads(init, "per_seed", half, checkpoint_dir=tmp_path)  # both seeds stop after 3 epochs, as if preempted
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t_s0.pt", "t_s1.pt"]
    resumed = heads(init, "per_seed", checkpoint_dir=tmp_path)
    for a, b in zip(straight, resumed):
        assert a.history == b.history and same(a, b)


def test_adoption_rule() -> None:
    def verdict(ci, f1_ci):
        return Verdict(ci, ci[1] < 0, f1_ci, not f1_ci[1] < 0, ci[1] < 0 and not f1_ci[1] < 0)

    assert adoption(verdict((-2.0, -0.5), (-0.01, 0.02)), -1.2) == "adopt"
    assert adoption(verdict((-2.0, 0.4), (-0.01, 0.02)), -0.8) == "borderline"
    assert adoption(verdict((-0.3, 1.5), (-0.01, 0.02)), 0.6) == "no"
    assert adoption(verdict((-2.0, -0.5), (-0.03, -0.01)), -1.2) == "no"  # SD lower but F1 worse: no claim
