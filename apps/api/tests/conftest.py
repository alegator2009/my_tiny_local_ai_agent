from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import settings


@pytest.fixture()
def isolated_data_dir(tmp_path: Path):
    prev_worker_enabled = settings.background_worker_enabled
    prev_data_dir = settings.data_dir
    prev_sqlite_path = settings.sqlite_path
    prev_lancedb_path = settings.lancedb_path
    prev_app_config_path = settings.app_config_path
    settings.background_worker_enabled = False
    settings.data_dir = str(tmp_path / "data")
    settings.sqlite_path = str(tmp_path / "data" / "app.db")
    settings.lancedb_path = str(tmp_path / "data" / "lancedb")
    settings.app_config_path = str(tmp_path / "data" / "config.json")
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    yield tmp_path
    settings.background_worker_enabled = prev_worker_enabled
    settings.data_dir = prev_data_dir
    settings.sqlite_path = prev_sqlite_path
    settings.lancedb_path = prev_lancedb_path
    settings.app_config_path = prev_app_config_path
