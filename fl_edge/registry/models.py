"""Registro de versiones de modelo (MODEL / MODEL_VERSION) + almacén de artefactos.

Cada versión registrada queda en disco con un layout fijo, relativo a
ARTIFACTS_DIR, que es el mismo que expone el servidor de artefactos por HTTP:

    ARTIFACTS_DIR/<model_name>/<version>/model.tflite
    ARTIFACTS_DIR/<model_name>/<version>/model.keras      (opcional)

En MODEL_VERSION se guardan rutas RELATIVAS a ARTIFACTS_DIR (portables entre
el host y los contenedores) y el SHA256 del .tflite, que el gestor de modelos
(Go) envía al dispositivo para verificar la integridad antes de activar.

Uso desde código (p. ej. al cerrar una ronda en server_app):

    from fl_edge.registry.models import register_version
    info = register_version(db_url, "artifacts", "fashion-cnn", "checkpoints/global_r3.tflite",
                            metrics={"central_acc": 0.87})

Uso por CLI:

    uv run python -m fl_edge.registry.models register \
        --model fashion-cnn --tflite checkpoints/global_r3.tflite \
        --keras checkpoints/global_r3.keras --metrics '{"central_acc": 0.87}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Mismas reglas que el model-manager del dispositivo y el gestor Go: el nombre
# del modelo termina siendo una carpeta en la Raspberry.
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TFLITE_NAME = "model.tflite"
KERAS_NAME = "model.keras"


def validate_model_name(name: str) -> None:
    if not MODEL_NAME_RE.match(name) or ".." in name:
        raise ValueError(
            f"Nombre de modelo inválido {name!r}: use letras, números, '.', '_' o '-' "
            "(máx. 64), sin espacios; se usa como carpeta en el dispositivo."
        )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_artifact_path(name: str, version: int, filename: str = TFLITE_NAME) -> str:
    """Ruta relativa a ARTIFACTS_DIR (con '/' siempre, también en Windows)."""
    return f"{name}/{version}/{filename}"


def stage_artifact(artifacts_dir: str | Path, rel_path: str, src: str | Path) -> Path:
    """Copia ``src`` a ``artifacts_dir/rel_path`` de forma atómica (tmp + rename),
    para que el servidor HTTP nunca sirva un archivo a medio copiar."""
    dst = Path(artifacts_dir) / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def db_url_from_env() -> str:
    """DATABASE_URL, o se arma con las mismas variables POSTGRES_* del resto
    de servicios (ver .env de iot-server-background)."""
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user = quote(os.getenv("POSTGRES_USER", "postgres-sbc-pms"), safe="")
    pwd = quote(os.getenv("POSTGRES_PASSWORD", "postgres-sbc-pms"), safe="")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "db_sbc_pms")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def register_version(
    db_url: str,
    artifacts_dir: str | Path,
    model_name: str,
    tflite_path: str | Path,
    keras_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    num_params: int | None = None,
    description: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Registra una nueva versión del modelo y devuelve
    ``{id_model, id_model_version, version, sha256, path_tflite, path_keras}``.

    La versión es MAX(VERSION)+1 del modelo, calculada con el MODEL bloqueado
    (SELECT ... FOR UPDATE) para que dos registros simultáneos no choquen.
    Si la inserción falla, se borran los archivos copiados.
    """
    import psycopg  # import diferido: el servidor de artefactos no lo necesita

    validate_model_name(model_name)
    tflite_path = Path(tflite_path)
    if not tflite_path.is_file():
        raise FileNotFoundError(f"No existe el .tflite: {tflite_path}")
    if keras_path is not None and not Path(keras_path).is_file():
        raise FileNotFoundError(f"No existe el .keras: {keras_path}")

    sha = sha256_file(tflite_path)
    staged: list[Path] = []

    # connect_timeout: si la BD no responde, fallar en segundos (no colgar el run)
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        try:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO MODEL (NAME, DESCRIPTION, TASK_TYPE)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (NAME) DO UPDATE SET
                        DESCRIPTION = COALESCE(EXCLUDED.DESCRIPTION, MODEL.DESCRIPTION),
                        TASK_TYPE   = COALESCE(EXCLUDED.TASK_TYPE, MODEL.TASK_TYPE)
                    RETURNING ID
                    """,
                    (model_name, description, task_type),
                )
                id_model = cur.fetchone()[0]
                cur.execute("SELECT ID FROM MODEL WHERE ID = %s FOR UPDATE", (id_model,))
                cur.execute(
                    "SELECT COALESCE(MAX(VERSION), 0) + 1 FROM MODEL_VERSION WHERE ID_MODEL = %s",
                    (id_model,),
                )
                version = cur.fetchone()[0]

                rel_tflite = relative_artifact_path(model_name, version, TFLITE_NAME)
                staged.append(stage_artifact(artifacts_dir, rel_tflite, tflite_path))
                rel_keras = None
                if keras_path is not None:
                    rel_keras = relative_artifact_path(model_name, version, KERAS_NAME)
                    staged.append(stage_artifact(artifacts_dir, rel_keras, keras_path))

                cur.execute(
                    """
                    INSERT INTO MODEL_VERSION
                        (ID_MODEL, VERSION, PATH_KERAS, PATH_TFLITE, SHA256, NUM_PARAMS, METRICS)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING ID
                    """,
                    (
                        id_model,
                        version,
                        rel_keras,
                        rel_tflite,
                        sha,
                        num_params,
                        json.dumps(metrics) if metrics is not None else None,
                    ),
                )
                id_model_version = cur.fetchone()[0]
        except Exception:
            for p in staged:
                p.unlink(missing_ok=True)
            raise

    return {
        "id_model": id_model,
        "id_model_version": id_model_version,
        "version": version,
        "sha256": sha,
        "path_tflite": rel_tflite,
        "path_keras": rel_keras,
    }


def try_register_version(
    model_name: str,
    tflite_path: str | Path | None,
    keras_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    num_params: int | None = None,
    task_type: str | None = "image_classification",
    db_url: str | None = None,
    artifacts_dir: str | Path | None = None,
    log=print,
) -> dict[str, Any] | None:
    """Versión tolerante a fallos de ``register_version`` para usar dentro del
    entrenamiento FL: NUNCA lanza excepción. Devuelve la info registrada, o
    ``None`` si no se pudo (y deja un aviso claro en el log).

    Usa DATABASE_URL / POSTGRES_* y ARTIFACTS_DIR del entorno si no se pasan.
    """
    if not tflite_path or not Path(tflite_path).is_file():
        log(f"[registry] AVISO: no hay .tflite final para registrar ({tflite_path}); se omite el registro")
        return None
    if keras_path is not None and not Path(keras_path).is_file():
        keras_path = None  # el .keras es opcional: si no está, se registra solo el .tflite
    try:
        info = register_version(
            db_url=db_url or db_url_from_env(),
            artifacts_dir=artifacts_dir or os.getenv("ARTIFACTS_DIR", "artifacts"),
            model_name=model_name,
            tflite_path=tflite_path,
            keras_path=keras_path,
            metrics=metrics,
            num_params=num_params,
            task_type=task_type,
        )
    except Exception as exc:  # noqa: BLE001 - el entrenamiento no debe caerse por esto
        log(f"[registry] AVISO: no se pudo registrar el modelo ({type(exc).__name__}: {exc}). "
            "El entrenamiento terminó bien; puedes registrarlo a mano con "
            "'python -m fl_edge.registry.models register'.")
        return None
    log(
        f"[registry] modelo registrado -> {model_name} v{info['version']} "
        f"(id_model_version={info['id_model_version']}, sha256={info['sha256'][:12]}...)"
    )
    return info


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fl_edge.registry.models", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="registra una nueva versión de un modelo")
    r.add_argument("--model", required=True, help="nombre del modelo (ej. fashion-cnn)")
    r.add_argument("--tflite", required=True, help="ruta al .tflite a desplegar")
    r.add_argument("--keras", help="ruta al .keras entrenable (opcional)")
    r.add_argument("--metrics", help="JSON con métricas de la versión (opcional)")
    r.add_argument("--description")
    r.add_argument("--task-type", default="image_classification")
    r.add_argument("--artifacts-dir", default=os.getenv("ARTIFACTS_DIR", "artifacts"))
    r.add_argument("--db-url", default=None, help="por defecto DATABASE_URL o POSTGRES_*")
    a = ap.parse_args(argv)

    info = register_version(
        db_url=a.db_url or db_url_from_env(),
        artifacts_dir=a.artifacts_dir,
        model_name=a.model,
        tflite_path=a.tflite,
        keras_path=a.keras,
        metrics=json.loads(a.metrics) if a.metrics else None,
        description=a.description,
        task_type=a.task_type,
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())