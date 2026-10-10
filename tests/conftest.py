import pytest

from ecgdelin.paths import ludb_root as _ludb_root

MARKERS = (
    "data: needs the LUDB download (skipped when absent)",
    "slow: long-running",
    "gpu: needs the real TimesFM model on a GPU (runs on Modal)",
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers here too, so the suite also runs inside the Modal image (no pyproject there)."""
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def ludb_root():
    root = _ludb_root()
    if not (root / "RECORDS").exists():
        pytest.skip(f"LUDB not found at {root} (set LUDB_DIR)")
    return root