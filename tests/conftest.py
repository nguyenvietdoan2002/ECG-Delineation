import pytest

from ecgdelin.paths import ludb_root as _ludb_root


@pytest.fixture
def ludb_root():
    root = _ludb_root()
    if not (root / "RECORDS").exists():
        pytest.skip(f"LUDB not found at {root} (set LUDB_DIR)")
    return root
