"""Where the data lives. Defaults assume the Meddies sibling-folder layout; env vars override."""

import os
from pathlib import Path

_MEDDIES = Path(__file__).resolve().parents[3]
DEFAULT_LUDB = (
    _MEDDIES / "20260926_Elsevier 2025" / "data" / "ludb"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
)
DEFAULT_TIMESFM_SRC = _MEDDIES / "timesfm" / "src"


def ludb_root() -> Path:
    return Path(os.environ.get("LUDB_DIR", DEFAULT_LUDB))


def timesfm_src() -> Path:
    return Path(os.environ.get("TIMESFM_SRC", DEFAULT_TIMESFM_SRC))
