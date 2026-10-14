"""Tests del registro de modelos y del servidor de artefactos.

    uv run pytest -v tests/test_registry.py
    # con BD de prueba (001 + 002 aplicados):
    PG_TEST_URL=postgresql://user:pass@localhost:5432/db uv run pytest -v tests/test_registry.py
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fl_edge.registry.artifact_server import make_server
from fl_edge.registry.models import (
    register_version,
    relative_artifact_path,
    sha256_file,
    stage_artifact,
    validate_model_name,
)


@pytest.fixture
def server(tmp_path):
    srv = make_server(tmp_path / "artifacts", host="127.0.0.1", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield tmp_path / "artifacts", f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_validate_model_name():
    validate_model_name("fashion-cnn")
    validate_model_name("cifar10_v2.small")
    for bad in ["Fashion CNN", "../etc", "a/b", "", "-x"]:
        with pytest.raises(ValueError):
            validate_model_name(bad)


def test_stage_and_serve_with_matching_sha(server, tmp_path):
    root, base = server
    src = tmp_path / "global_r3.tflite"
    src.write_bytes(b"tflite bytes de prueba")
    rel = relative_artifact_path("fashion-cnn", 3)
    assert rel == "fashion-cnn/3/model.tflite"
    stage_artifact(root, rel, src)

    status, body = _get(f"{base}/artifacts/{rel}")
    assert status == 200
    # lo que baja el dispositivo coincide con el sha que se guarda en la BD
    assert hashlib.sha256(body).hexdigest() == sha256_file(src)
    assert _get(f"{base}/healthz")[0] == 200


@pytest.mark.parametrize(
    "path",
    [
        "/artifacts/fashion-cnn/9/model.tflite",  # no existe
        "/artifacts/fashion-cnn/3/model.keras",  # solo se sirve el .tflite
        "/artifacts/../../etc/passwd",
        "/artifacts/fashion-cnn/",  # sin listado
        "/artifacts/fashion-cnn/0/model.tflite",
        "/",
    ],
)
def test_server_rejects(server, tmp_path, path):
    root, base = server
    src = tmp_path / "m.tflite"
    src.write_bytes(b"x")
    stage_artifact(root, "fashion-cnn/3/model.tflite", src)
    stage_artifact(root, "fashion-cnn/3/model.keras", src)
    assert _get(base + path)[0] == 404


@pytest.mark.skipif(not os.getenv("PG_TEST_URL"), reason="PG_TEST_URL no definido")
def test_register_version_increments_and_persists(tmp_path):
    import psycopg

    url = os.environ["PG_TEST_URL"]
    name = "itest-registry"
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DELETE FROM MODEL WHERE NAME = %s", (name,))

    src = tmp_path / "g.tflite"
    arts = tmp_path / "artifacts"
    try:
        src.write_bytes(b"v1")
        a = register_version(url, arts, name, src, metrics={"central_acc": 0.8})
        src.write_bytes(b"v2")
        b = register_version(url, arts, name, src)

        assert (a["version"], b["version"]) == (1, 2)
        assert a["path_tflite"] == f"{name}/1/model.tflite"
        assert (arts / b["path_tflite"]).read_bytes() == b"v2"
        assert b["sha256"] == hashlib.sha256(b"v2").hexdigest()

        with psycopg.connect(url) as c:
            row = c.execute(
                "SELECT VERSION, PATH_TFLITE, SHA256, METRICS FROM MODEL_VERSION WHERE ID = %s",
                (a["id_model_version"],),
            ).fetchone()
        assert row[0] == 1 and row[1] == a["path_tflite"] and row[2] == a["sha256"]
        assert row[3] == {"central_acc": 0.8}
    finally:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("DELETE FROM MODEL WHERE NAME = %s", (name,))


@pytest.mark.skipif(not os.getenv("PG_TEST_URL"), reason="PG_TEST_URL no definido")
def test_register_rejects_bad_name_without_touching_disk(tmp_path):
    src = tmp_path / "g.tflite"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        register_version(os.environ["PG_TEST_URL"], tmp_path / "a", "Bad Name", src)
    assert not (tmp_path / "a").exists()
