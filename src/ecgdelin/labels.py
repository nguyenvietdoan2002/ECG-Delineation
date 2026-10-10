"""Per-sample training labels from the cardiologists' marks."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ecgdelin.dataset import Wave

CLASSES = ("none", "P", "QRS", "T")


@dataclass(frozen=True)
class Labels:
    classes: NDArray[np.int8]  # index into CLASSES, per sample
    mask: NDArray[np.bool_]  # True where the sample counts for training and scoring


def labels_for(waves: Sequence[Wave], n_samples: int) -> Labels:
    """Label onset..offset (inclusive) of every complete wave.

    Usable samples run from the first mark to the last mark of the lead, except around incomplete
    waves (a peak missing its onset or offset): the whole gap between the neighbouring complete
    waves is masked out, because the incomplete wave's true extent is unknown.
    """
    classes = np.zeros(n_samples, dtype=np.int8)
    mask = np.zeros(n_samples, dtype=bool)
    if not waves:
        return Labels(classes, mask)

    marks = [s for w in waves for s in (w.onset, w.peak, w.offset) if s is not None]
    mask[min(marks) : max(marks) + 1] = True

    complete = [w for w in waves if w.onset is not None and w.offset is not None]
    for wave in complete:
        assert wave.onset is not None and wave.offset is not None
        classes[wave.onset : wave.offset + 1] = CLASSES.index(wave.kind)

    for wave in waves:
        if wave.onset is not None and wave.offset is not None:
            continue
        before = [w.offset for w in complete if w.offset is not None and w.offset < wave.peak]
        after = [w.onset for w in complete if w.onset is not None and w.onset > wave.peak]
        start = max(before) + 1 if before else 0
        stop = min(after) if after else n_samples
        mask[start:stop] = False
    return Labels(classes, mask)
