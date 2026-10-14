"""Servidor HTTP de artefactos de modelo (solo librería estándar).

Sirve ARTIFACTS_DIR en modo de solo lectura para que el model-manager de cada
dispositivo descargue el .tflite que le indica el gestor de modelos:

    GET|HEAD /artifacts/<model_name>/<version>/model.tflite
    GET      /healthz

Decisiones:
- Solo se aceptan rutas con ese patrón exacto (sin listado de directorios ni
  ``..``), de modo que no expone nada más del disco del servidor.
- Sin dependencias: se puede correr en ``python:3.12-alpine`` montando solo
  este archivo, o desde el entorno uv de iot-server-fl.
- La integridad no depende de este servidor (HTTP plano): el dispositivo
  verifica el sha256 que llega por MQTT/TLS antes de activar. En la nube se
  puede reemplazar por un bucket (GCS) sin tocar la Pi: solo cambia
  ARTIFACT_BASE_URL en el gestor de modelos.

Uso:
    ARTIFACTS_DIR=artifacts ARTIFACT_PORT=8000 uv run python -m fl_edge.registry.artifact_server
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROUTE_RE = re.compile(
    r"^/artifacts/(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,63})/(?P<version>[1-9][0-9]{0,8})/"
    r"(?P<file>model\.tflite)$"
)

log = logging.getLogger("artifact_server")


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "fl-artifacts/1.0"

        def _resolve(self) -> Path | None:
            path = self.path.split("?", 1)[0]
            m = ROUTE_RE.match(path)
            if not m or ".." in m["name"]:
                return None
            target = (root / m["name"] / m["version"] / m["file"]).resolve()
            if root not in target.parents or not target.is_file():
                return None
            return target

        def _send_file(self, with_body: bool) -> None:
            if self.path == "/healthz":
                body = b"ok\n"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if with_body:
                    self.wfile.write(body)
                return
            target = self._resolve()
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = target.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if with_body:
                with open(target, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)

        def do_GET(self) -> None:  # noqa: N802
            self._send_file(with_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._send_file(with_body=False)

        def log_message(self, fmt: str, *args) -> None:
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def make_server(root: str | Path, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), make_handler(root))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = os.getenv("ARTIFACTS_DIR", "artifacts")
    port = int(os.getenv("ARTIFACT_PORT", "8000"))
    srv = make_server(root, port=port)
    log.info("Sirviendo %s en :%d (GET /artifacts/<model>/<version>/model.tflite)", Path(root).resolve(), port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
