"""Build and save the LUDB person-level folds. Deterministic; run once and commit the JSON.

    uv run python scripts/make_splits.py --seed 0
"""

import argparse
from collections import Counter
from pathlib import Path

from ecgdelin.dataset import load_record, record_ids
from ecgdelin.paths import ludb_root
from ecgdelin.splits import make_folds, rhythm_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: splits/ludb_5fold_seed<seed>.json")
    args = parser.parse_args()
    out = args.out or Path("splits") / f"ludb_5fold_seed{args.seed}.json"

    root = ludb_root()
    ids = record_ids(root)
    groups = {rid: rhythm_group(load_record(root, rid).rhythm) for rid in ids}
    folds = make_folds(ids, groups, seed=args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(folds.to_json(), encoding="utf-8")
    print(f"{len(ids)} people, groups {dict(Counter(groups.values()))} -> {out}")
    for k, fold in enumerate(folds.folds):
        test_groups = Counter(groups[p] for p in fold.test)
        print(
            f"fold {k}: test {len(fold.test)} {dict(test_groups)}, val {len(fold.val)}, "
            f"train {len(fold.train)}, subsets "
            + ", ".join(f"{frac:.0%}={len(people)}" for frac, people in fold.subsets.items())
        )


if __name__ == "__main__":
    main()
