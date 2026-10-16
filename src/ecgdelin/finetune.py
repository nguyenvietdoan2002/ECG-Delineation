"""Arm B of the PTB-XL+ experiment (decisions 72 and 78): step 1's head started from pretrained weights
and trained on one LUDB fold with step 1's recipe, one model per seed, and the adoption rule.

Step 1's seeds differed only in the head's random start, over one data stream per fold (decision 31).
Started from one pretrained head, such seeds would be the same run three times, so ``finetune_heads``
offers both ways for seeds to differ: ``streams="per_seed"`` gives each seed its own shuffling (seed 0's
is step 1's stream) from whatever start it is given; ``streams="shared"`` keeps step 1's single stream,
which only makes sense when the seeds start from different pretrained heads.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from ecgdelin.score import Verdict
from ecgdelin.train import Source, TrainConfig, Trainee, make_trainee, train_jointly

STREAMS = ("per_seed", "shared")


def finetune_heads(
    make_model: Callable[[], nn.Module],
    init: Mapping[int, Mapping[str, Tensor]],
    sources: Mapping[str, Source],
    targets: Tensor,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    config: TrainConfig,
    device: torch.device,
    streams: str = "per_seed",
    checkpoint_dir: Path | None = None,
    run: str = "finetune",
    log: Callable[[str], None] = print,
    after_epoch: Callable[[int], None] | None = None,
) -> list[Trainee]:
    """One head per seed in ``init`` (seed -> pretrained state dict), each named ``head_s<seed>`` and
    ending with its best weights. Checkpoints go to ``<checkpoint_dir>/<run>_s<seed>.pt`` (per-seed
    streams) or ``<checkpoint_dir>/<run>.pt`` (shared), so a preempted container resumes."""
    if streams not in STREAMS:
        raise ValueError(f"streams must be one of {STREAMS}, got {streams!r}")
    seeds = sorted(init)
    trainees = []
    for seed in seeds:
        model = make_model()
        model.load_state_dict(init[seed])
        trainees.append(make_trainee(f"head_s{seed}", model, "states", config, normalized=True))

    def ckpt(name: str) -> Path | None:
        return None if checkpoint_dir is None else checkpoint_dir / f"{name}.pt"

    if streams == "shared":
        train_jointly(trainees, sources, targets, train_idx, val_idx, config, seeds[0], device, checkpoint=ckpt(run), log=log, after_epoch=after_epoch)
    else:
        for seed, t in zip(seeds, trainees):
            train_jointly([t], sources, targets, train_idx, val_idx, config, seed, device, checkpoint=ckpt(f"{run}_s{seed}"), log=log, after_epoch=after_epoch)
    return trainees


def adoption(verdict: Verdict, point_diff_ms: float) -> str:
    """Decision 78 for arm B against the step-1 head: ``adopt`` when the paired 95% interval of the
    primary-SD difference lies below 0 and the F1 interval does not; ``borderline`` when the point
    estimate favours pretraining but the interval reaches 0 (two more pretraining seeds run first);
    otherwise ``no``."""
    if verdict.a_wins:
        return "adopt"
    if point_diff_ms < 0 and not verdict.a_better:
        return "borderline"
    return "no"
