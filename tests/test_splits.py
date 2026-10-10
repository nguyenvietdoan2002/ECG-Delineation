"""Splits seam: make_folds(people, groups, seed) -> person-level, rhythm-stratified folds."""

import pytest

from ecgdelin.splits import Folds, make_folds, rhythm_group

PEOPLE = [f"p{i}" for i in range(40)]
GROUPS = {p: ("af_flutter" if i % 5 == 0 else "sinus") for i, p in enumerate(PEOPLE)}  # 8 AF, 32 sinus


def test_every_person_tested_exactly_once_and_af_spread_across_folds():
    folds = make_folds(PEOPLE, GROUPS, seed=0)
    assert len(folds.folds) == 5
    tested = [p for fold in folds.folds for p in fold.test]
    assert sorted(tested) == sorted(PEOPLE)
    for fold in folds.folds:
        assert len(fold.test) == 8
        assert 1 <= sum(GROUPS[p] == "af_flutter" for p in fold.test) <= 2


def test_validation_is_a_fifth_of_the_training_fold_and_disjoint_from_test():
    folds = make_folds(PEOPLE, GROUPS, seed=0)
    for fold in folds.folds:
        assert sorted(fold.test + fold.val + fold.train) == sorted(PEOPLE)
        assert len(fold.val) == 6  # 20% of the 32 training people, stratified
        assert any(GROUPS[p] == "af_flutter" for p in fold.val)


def test_learning_curve_subsets_are_nested_and_stratified():
    folds = make_folds(PEOPLE, GROUPS, seed=0)
    for fold in folds.folds:
        quarter, half = fold.subsets[0.25], fold.subsets[0.5]
        assert set(quarter) < set(half) < set(fold.train)
        assert 5 <= len(quarter) <= 8 and 11 <= len(half) <= 15  # ~26 training people
        assert any(GROUPS[p] == "af_flutter" for p in quarter)


def test_folds_are_deterministic_per_seed_and_survive_json():
    assert make_folds(PEOPLE, GROUPS, seed=0) == make_folds(PEOPLE, GROUPS, seed=0)
    assert make_folds(PEOPLE, GROUPS, seed=0) != make_folds(PEOPLE, GROUPS, seed=1)
    folds = make_folds(PEOPLE, GROUPS, seed=3)
    assert Folds.from_json(folds.to_json()) == folds


@pytest.mark.parametrize(
    ("rhythm", "group"),
    [
        ("Sinus rhythm", "sinus"),
        ("Sinus bradycardia", "sinus"),
        ("Irregular sinus rhythm", "sinus"),
        ("Atrial fibrillation", "af_flutter"),
        ("Atrial flutter, typical", "af_flutter"),
        ("Junctional rhythm", "other"),
    ],
)
def test_rhythm_group(rhythm, group):
    assert rhythm_group(rhythm) == group
