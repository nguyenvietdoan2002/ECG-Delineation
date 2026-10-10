"""Labels seam: labels_for(waves, n_samples) -> per-sample classes and the mask of usable samples."""

import numpy as np

from ecgdelin.dataset import Wave
from ecgdelin.labels import CLASSES, labels_for


def test_complete_waves_label_their_samples_and_mask_the_marked_span():
    waves = [Wave("P", 10, 14, 18), Wave("QRS", 30, 33, 36), Wave("T", 50, 60, 70)]
    labels = labels_for(waves, n_samples=100)

    assert CLASSES == ("none", "P", "QRS", "T")
    expected = np.zeros(100, dtype=np.int8)
    expected[10:19] = CLASSES.index("P")  # onset and offset samples belong to the wave
    expected[30:37] = CLASSES.index("QRS")
    expected[50:71] = CLASSES.index("T")
    assert np.array_equal(labels.classes, expected)

    expected_mask = np.zeros(100, dtype=bool)
    expected_mask[10:71] = True  # first mark to last mark, inclusive
    assert np.array_equal(labels.mask, expected_mask)


def test_incomplete_waves_are_unlabelled_and_their_surroundings_masked_out():
    waves = [
        Wave("QRS", 30, 33, 36),
        Wave("T", None, 60, 70),  # no onset
        Wave("P", 90, 95, None),  # no offset
        Wave("QRS", 110, 113, 116),
    ]
    labels = labels_for(waves, n_samples=130)

    expected = np.zeros(130, dtype=np.int8)
    expected[30:37] = CLASSES.index("QRS")
    expected[110:117] = CLASSES.index("QRS")
    assert np.array_equal(labels.classes, expected)

    expected_mask = np.zeros(130, dtype=bool)
    expected_mask[30:37] = True
    expected_mask[110:117] = True  # the gap between the complete waves is unknown territory
    assert np.array_equal(labels.mask, expected_mask)
