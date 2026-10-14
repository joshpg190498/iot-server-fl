"""Registro de modelos y artefactos (MLOps) del servidor FL.

- ``models``: registra una versión (MODEL / MODEL_VERSION), calcula su sha256 y
  la copia al almacén de artefactos con el layout ``<name>/<version>/model.tflite``.
- ``artifact_server``: sirve ese almacén por HTTP para que el model-manager del
  dispositivo descargue el ``.tflite``.
"""
