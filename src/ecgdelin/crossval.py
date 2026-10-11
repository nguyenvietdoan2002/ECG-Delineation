"""Cross-validated predictions: every person is predicted by the models of the fold that held them out,
and the folds' test predictions are pooled into one prediction set over all people (decision 64)."""

from collections.abc import Mapping, Sequence

import numpy as np

from ecgdelin.dataset import Recording
from ecgdelin.delineate import LeadPoints
from ecgdelin.score import Predictions, score
from ecgdelin.splits import Folds


def assemble(folds: Folds, by_fold: Mapping[int, Predictions], only: Sequence[int] | None = None) -> dict[str, Mapping[str, LeadPoints]]:
    """One prediction set from per-fold test predictions. Every test person of every fold in ``only``
    (default: all folds) appears exactly once; anything missing, duplicated or outside a fold's test
    set is an error, because a silent gap would bias the pooled metric."""
    wanted = list(range(len(folds.folds))) if only is None else list(only)
    out: dict[str, Mapping[str, LeadPoints]] = {}
    for k in wanted:
        if k not in by_fold:
            raise ValueError(f"fold {k}: no predictions")
        preds = by_fold[k]
        test = folds.folds[k].test
        extra = sorted(set(preds) - set(test))
        if extra:
            raise ValueError(f"fold {k}: predictions for people outside its test set: {extra[:5]}")
        for rec_id in test:
            if rec_id not in preds:
                raise ValueError(f"fold {k}: no prediction for test person {rec_id}")
            if rec_id in out:
                raise ValueError(f"person {rec_id} predicted by more than one fold")
            out[rec_id] = preds[rec_id]
    return out


def per_fold_primary(
    recordings: Mapping[str, Recording], folds: Folds, by_fold_seeds: Mapping[int, Sequence[Predictions]], only: Sequence[int] | None = None
) -> list[float]:
    """Per fold, the primary metric on that fold's test people averaged over the seeds given for it;
    reported for information only (decision 64)."""
    wanted = list(range(len(folds.folds))) if only is None else list(only)
    out = []
    for k in wanted:
        recs = [recordings[i] for i in folds.folds[k].test]
        out.append(float(np.mean([score(recs, preds, n_draws=0).primary_sd_ms for preds in by_fold_seeds[k]])))
    return out
