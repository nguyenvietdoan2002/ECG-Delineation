"""Dataset seam: load_record(root, id) -> Recording. Literal values come from the raw LUDB files."""

import numpy as np
import pytest

from ecgdelin.dataset import LEADS, Wave, load_record, record_ids, waves_from_annotations


@pytest.mark.data
def test_record_ids_are_the_200_ludb_records(ludb_root):
    ids = record_ids(ludb_root)
    assert len(ids) == 200
    assert set(ids) == {str(i) for i in range(1, 201)}


@pytest.mark.data
def test_record_1_signal_and_header(ludb_root):
    rec = load_record(ludb_root, "1")
    assert rec.id == "1"
    assert rec.fs == 500
    assert rec.signal.shape == (12, 5000)
    assert rec.signal.dtype == np.float64
    assert LEADS == ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
    # 1.hea, lead ii: "1.dat 16 1206(2)/mV 0 0 25 ..." -> gain 1206, baseline 2, first raw value 25
    assert rec.lead("ii")[0] == pytest.approx((25 - 2) / 1206)
    assert np.array_equal(rec.lead("ii"), rec.signal[1])
    assert rec.rhythm == "Sinus bradycardia"


@pytest.mark.data
def test_record_1_lead_ii_marks(ludb_root):
    rec = load_record(ludb_root, "1")
    # 1.ii annotation file: ( N ) ( t ) ( p ) ...
    assert rec.marks["ii"][:3] == [
        Wave("QRS", 644, 662, 682),
        Wave("T", 776, 843, 878),
        Wave("P", 1250, 1278, 1302),
    ]
    assert set(rec.marks) == set(LEADS)


def test_waves_from_annotations_keeps_incomplete_waves_with_missing_sides():
    symbols = ["(", "N", ")", "t", ")", "(", "p"]
    samples = [10, 12, 15, 40, 50, 70, 75]
    assert waves_from_annotations(symbols, samples) == [
        Wave("QRS", 10, 12, 15),
        Wave("T", None, 40, 50),
        Wave("P", 70, 75, None),
    ]
