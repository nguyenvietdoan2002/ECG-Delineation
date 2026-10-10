"""Person-level, rhythm-stratified cross-validation folds, with validation and learning-curve subsets."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

GROUPS = ("sinus", "af_flutter", "other")


def rhythm_group(rhythm: str) -> str:
    """Coarse rhythm class used for stratification and result breakdowns."""
    lowered = rhythm.lower()
    if "sinus" in lowered:
        return "sinus"
    if "atrial fibrillation" in lowered or "atrial flutter" in lowered:
        return "af_flutter"
    return "other"


@dataclass(frozen=True)
class Fold:
    test: tuple[str, ...]
    val: tuple[str, ...]  # early stopping only; carved from the training people
    train: tuple[str, ...]  # the 100% training set (training fold minus val)
    subsets: dict[float, tuple[str, ...]]  # fraction -> nested subset of train (25% within 50%)


@dataclass(frozen=True)
class Folds:
    seed: int
    folds: tuple[Fold, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "folds": [
                    {
                        "test": list(f.test),
                        "val": list(f.val),
                        "train": list(f.train),
                        "subsets": {str(frac): list(people) for frac, people in f.subsets.items()},
                    }
                    for f in self.folds
                ],
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> "Folds":
        data = json.loads(text)
        return cls(
            seed=data["seed"],
            folds=tuple(
                Fold(
                    test=tuple(f["test"]),
                    val=tuple(f["val"]),
                    train=tuple(f["train"]),
                    subsets={float(frac): tuple(people) for frac, people in f["subsets"].items()},
                )
                for f in data["folds"]
            ),
        )


def _by_group(people: Sequence[str], groups: Mapping[str, str]) -> dict[str, list[str]]:
    present = sorted({groups[p] for p in people})
    return {g: sorted(p for p in people if groups[p] == g) for g in present}


def _stratified_prefix(
    people: Sequence[str], groups: Mapping[str, str], frac: float, rng: np.random.Generator
) -> list[list[str]]:
    """Each group shuffled once; a prefix of each is taken, so smaller fractions nest in larger ones."""
    return [list(rng.permutation(members)) for members in _by_group(people, groups).values()]


def _take(order: list[list[str]], frac: float) -> tuple[str, ...]:
    chosen = []
    for members in order:
        count = max(1, int(frac * len(members) + 0.5)) if members else 0
        chosen.extend(members[:count])
    return tuple(chosen)


def make_folds(
    people: Sequence[str],
    groups: Mapping[str, str],
    seed: int,
    n_folds: int = 5,
    val_frac: float = 0.2,
    subset_fracs: Sequence[float] = (0.25, 0.5),
) -> Folds:
    rng = np.random.default_rng(seed)
    # Deal each rhythm group round-robin into the folds with one running counter, so every fold gets
    # its share of each group and the fold sizes stay balanced.
    test_sets: list[list[str]] = [[] for _ in range(n_folds)]
    counter = int(rng.integers(n_folds))
    for members in _by_group(people, groups).values():
        for person in rng.permutation(members):
            test_sets[counter % n_folds].append(str(person))
            counter += 1

    folds = []
    for test in test_sets:
        rest = [p for p in people if p not in set(test)]
        val = _take(_stratified_prefix(rest, groups, val_frac, rng), val_frac)
        train = [p for p in rest if p not in set(val)]
        order = _stratified_prefix(train, groups, 1.0, rng)
        subsets = {frac: _take(order, frac) for frac in subset_fracs}
        folds.append(Fold(tuple(test), val, tuple(train), subsets))
    return Folds(seed=seed, folds=tuple(folds))
