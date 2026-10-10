"""LUDB records: 12-lead signals in mV, header rhythm, and the cardiologists' per-lead wave marks."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import wfdb
from numpy.typing import NDArray

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
PEAK_SYMBOLS = {"p": "P", "N": "QRS", "t": "T"}  # WFDB annotation symbol -> wave


class Wave(NamedTuple):
    """One marked wave. ``onset``/``offset`` are None when the cardiologists left that side unmarked."""

    kind: str  # "P", "QRS" or "T"
    onset: int | None
    peak: int
    offset: int | None


@dataclass(frozen=True)
class Recording:
    id: str
    fs: int
    signal: NDArray[np.float64]  # (12, samples) in mV, rows in LEADS order
    rhythm: str  # the header's "Rhythm:" line, without the trailing full stop
    marks: dict[str, list[Wave]]  # lead name -> waves in time order

    def lead(self, name: str) -> NDArray[np.float64]:
        return self.signal[LEADS.index(name)]


def waves_from_annotations(symbols: Sequence[str], samples: Sequence[int]) -> list[Wave]:
    """Group WFDB symbols into waves: a peak takes the '(' just before it and the ')' just after."""
    waves = []
    for k, (symbol, sample) in enumerate(zip(symbols, samples)):
        if symbol not in PEAK_SYMBOLS:
            continue
        onset = int(samples[k - 1]) if k > 0 and symbols[k - 1] == "(" else None
        offset = int(samples[k + 1]) if k + 1 < len(symbols) and symbols[k + 1] == ")" else None
        waves.append(Wave(PEAK_SYMBOLS[symbol], onset, int(sample), offset))
    return waves


def record_ids(root: Path) -> list[str]:
    """Record ids from the RECORDS file (its lines look like ``data/1``)."""
    lines = (Path(root) / "RECORDS").read_text(encoding="utf-8").split()
    return [line.rsplit("/", 1)[-1] for line in lines]


def load_record(root: Path, record_id: str) -> Recording:
    path = str(Path(root) / "data" / record_id)
    record = wfdb.rdrecord(path)
    names = [name.lower() for name in record.sig_name]
    signal = np.ascontiguousarray(record.p_signal.T[[names.index(lead) for lead in LEADS]])
    rhythm = ""
    for comment in record.comments:
        if comment.strip().startswith("Rhythm:"):
            rhythm = comment.split(":", 1)[1].strip().rstrip(".")
    marks = {}
    for lead in LEADS:
        annotation = wfdb.rdann(path, lead)
        marks[lead] = waves_from_annotations(list(annotation.symbol), annotation.sample.tolist())
    return Recording(record_id, int(record.fs), signal.astype(np.float64), rhythm, marks)
