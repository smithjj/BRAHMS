import os
import sys
from pathlib import Path

import pytest

# Qt widgets must run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui.core_py import crystal_db  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """A fresh, isolated CrystalDB (seeded with the preloaded crystals)."""
    return crystal_db.CrystalDB(tmp_path / "crystals.db")


@pytest.fixture(autouse=True)
def isolated_global_db(tmp_path_factory, monkeypatch):
    """Anything that calls get_db() gets a throwaway DB instead of the
    user's real crystals/crystals.db."""
    path = tmp_path_factory.mktemp("globaldb") / "crystals.db"
    monkeypatch.setattr(crystal_db, "_instance", crystal_db.CrystalDB(path))


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
