
# iot-server-fl — runtime de aprendizaje federado (Flower, Python)
FROM python:3.11-slim
 
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
 
WORKDIR /app
 
# El entorno virtual vive FUERA de /app para poder bind-montar el código en
# desarrollo (`.:/app`) sin que el montaje del host oculte el venv del contenedor.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
 
# 1) Metadatos + paquete primero, para cachear la capa de dependencias
COPY pyproject.toml README.md ./
COPY fl_edge/ ./fl_edge/
 
# 2) Instalar dependencias (Flower + TensorFlow-CPU + Ray para la simulación)
RUN uv sync
 
# 3) Resto del proyecto (db/, etc.)
COPY . .
 
# Por defecto: simulación local de M0 (requiere server_app/client_app, pasos 3-4).
CMD ["uv", "run", "flwr", "run", ".", "local-sim", "--stream"]