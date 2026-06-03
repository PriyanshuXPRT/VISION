"""Tests for the SQLite repository layer (in-memory DB)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from vision.database import (
    AuthLogRepository,
    Database,
    DeviceRepository,
    FaceTemplateRepository,
    UserRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    p = tmp_path / "test.db"
    d = Database(p)
    yield d
    d.close()


def test_user_crud(db: Database):
    users = UserRepository(db)
    u = users.create(name="Alice", phone="+1", email="a@x.com")
    assert u.user_id > 0
    assert users.get(u.user_id).name == "Alice"
    users.update(u.user_id, phone="+2")
    assert users.get(u.user_id).phone == "+2"
    assert len(users.list()) == 1
    users.delete(u.user_id)
    with pytest.raises(Exception):
        users.get(u.user_id)


def test_template_storage(db: Database):
    users = UserRepository(db)
    tpls = FaceTemplateRepository(db)
    u = users.create(name="Bob")
    emb = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    t = tpls.add(u.user_id, emb, 0.7, source="admin")
    out = tpls.get(t.template_id)
    assert out.embedding.shape == (512,)
    assert np.allclose(out.embedding, emb, atol=1e-5)


def test_auth_log(db: Database):
    logs = AuthLogRepository(db)
    logs.add(user_id=None, liveness_score=0.9, similarity_score=0.3, result="reject", reason="liveness")
    logs.add(user_id=None, liveness_score=0.2, similarity_score=0.9, result="reject", reason="spoof")
    assert len(logs.list_recent(limit=10)) == 2


def test_device_registry(db: Database):
    devs = DeviceRepository(db)
    devs.register("dev-1", "Pixel 7", "android")
    devs.touch("dev-1")
    assert devs.get("dev-1").device_name == "Pixel 7"
    assert len(devs.list()) == 1
