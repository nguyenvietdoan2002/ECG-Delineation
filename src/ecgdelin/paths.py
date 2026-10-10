"""Where the data lives. Defaults assume the Meddies sibling-folder layout; env vars override."""

import os
from pathlib import Path

def _meddies_root() -> Path:
    """The Meddies folder, three levels above this package. Falls back to the highest existing
    parent when the package sits shallower, as it does at ``/root/ecgdelin`` in the Modal image."""
    parents = Path(__file__).resolve().parents
    return parents[min(3, len(parents) - 1)]


_MEDDIES = _meddies_root()
DEFAULT_LUDB = (
    _MEDDIES / "20260926_Elsevier 2025" / "data" / "ludb"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
)
DEFAULT_TIMESFM_SRC = _MEDDIES / "timesfm" / "src"


def ludb_root() -> Path:
    return Path(os.environ.get("LUDB_DIR", DEFAULT_LUDB))


def timesfm_src() -> Path:
    return Path(os.environ.get("TIMESFM_SRC", DEFAULT_TIMESFM_SRC))
