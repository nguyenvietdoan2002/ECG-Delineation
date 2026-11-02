"""Where the data lives. Defaults assume LUDB in ``data/ludb`` inside the repo and the TimesFM checkout
next to the repo (``../timesfm``); env vars override."""

import os
from pathlib import Path


def _repo_root() -> Path:
    """The repo, two levels above this package. Falls back to the highest existing parent when the
    package sits shallower, as it does at ``/root/ecgdelin`` in the Modal image (where the env vars
    are set anyway)."""
    parents = Path(__file__).resolve().parents
    return parents[min(2, len(parents) - 1)]


_REPO = _repo_root()
DEFAULT_LUDB = _REPO / "data" / "ludb"
DEFAULT_TIMESFM_SRC = _REPO.parent / "timesfm" / "src"


def ludb_root() -> Path:
    return Path(os.environ.get("LUDB_DIR", DEFAULT_LUDB))


def timesfm_src() -> Path:
    return Path(os.environ.get("TIMESFM_SRC", DEFAULT_TIMESFM_SRC))
