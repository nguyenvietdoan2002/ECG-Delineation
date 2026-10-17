"""RDB marks and signals: parsing, decision 75's rules on the shared marks, units, the stopping split."""

import numpy as np

from ecgdelin.dataset import LEADS
from ecgdelin.labels import labels_for
from ecgdelin.rdb import parse_marks, rdb_waves, read_signal, rhythm_class, stop_split

CLEAN = "﻿TYPE,START,END\r\n1,467,513\r\n0,400,440\r\n2,560,700\r\n0,900,940\r\n1,967,1019\r\n2,1060,1200\r\n"


def test_parse_marks_skips_header_and_bom_and_sorts_by_start() -> None:
    marks = parse_marks(CLEAN)
    assert marks == [("P", 400.0, 440.0), ("QRS", 467.0, 513.0), ("T", 560.0, 700.0), ("P", 900.0, 940.0), ("QRS", 967.0, 1019.0), ("T", 1060.0, 1200.0)]
    assert parse_marks("0,10.5,20.25\n") == [("P", 10.5, 20.25)]


def test_clean_beats_label_their_spans_and_nothing_else() -> None:
    lw = rdb_waves(parse_marks(CLEAN))
    assert lw.n_beats == 2 and lw.n_masked_beats == 0 and lw.aligned
    assert all(w.onset is not None and w.offset is not None for w in lw.waves)
    labels = labels_for(lw.waves, 5000)
    assert labels.classes[400:441].tolist() == [1] * 41 and labels.classes[467:514].tolist() == [2] * 47 and labels.classes[560:701].tolist() == [3] * 141
    assert labels.classes[441:467].tolist() == [0] * 26 and labels.mask[400:1201].all() and not labels.mask[:400].any() and not labels.mask[1201:].any()


def test_overlapping_spans_are_masked_with_their_neighbour() -> None:
    marks = parse_marks("1,467,513\n2,560,920\n0,900,940\n1,967,1019\n")  # the T runs into the next P
    lw = rdb_waves(marks)
    kinds = {(w.kind, w.peak): w for w in lw.waves}
    assert lw.n_masked_beats == 2
    assert kinds[("T", 740)].onset is None and kinds[("T", 740)].offset is None
    assert kinds[("P", 920)].onset is None and kinds[("P", 920)].offset is None
    assert kinds[("QRS", 490)].offset == 513 and kinds[("QRS", 993)].onset == 967
    labels = labels_for(lw.waves, 5000)
    assert not labels.mask[514:967].any() and labels.mask[467:514].all() and labels.mask[967:1020].all()


def test_edge_marks_count_as_missing_and_fractions_are_rounded() -> None:
    lw = rdb_waves(parse_marks("1,0,30\n2,60.4,4967.6\n0,4990,5007.8\n"))
    by = {w.kind: w for w in lw.waves}
    assert by["QRS"].onset is None and by["QRS"].offset == 30 and by["QRS"].peak == 15
    assert by["T"].onset == 60 and by["T"].offset == 4968
    assert by["P"].onset == 4990 and by["P"].offset is None and by["P"].peak == 4999  # ends beyond the record: offset unknown
    assert rdb_waves([("QRS", 100.0, 90.0)]).n_masked_beats == 1  # backwards span


def test_signal_is_microvolts_reordered_to_leads() -> None:
    rows = [[1000 * (c + 1) for c in range(12)] for _ in range(4)]  # column c holds (c + 1) mV in microvolts
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    sig = read_signal(text)
    assert sig.shape == (12, 4) and sig.dtype == np.float32
    assert np.allclose(sig[LEADS.index("v6")], 12.0) and np.allclose(sig[LEADS.index("i")], 1.0)


def test_stop_split_holds_out_the_same_share_of_every_class() -> None:
    ids = [f"AF{k:04d}" for k in range(100)] + [f"SR{k:04d}" for k in range(50)] + [f"VT{k:04d}" for k in range(9)]
    train, stop = stop_split(ids, 0.1, seed=0)
    assert sorted(train + stop) == sorted(ids) and not set(train) & set(stop)
    assert {rhythm_class(i) for i in stop} == {"AF", "SR", "VT"}
    assert sum(i.startswith("AF") for i in stop) == 10 and sum(i.startswith("SR") for i in stop) == 5 and sum(i.startswith("VT") for i in stop) == 1
    assert stop == stop_split(ids, 0.1, seed=0)[1] and stop != stop_split(ids, 0.1, seed=1)[1]
