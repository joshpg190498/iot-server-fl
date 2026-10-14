# iot-server-fl — núcleo de aprendizaje federado (Sprint 1 · M0)

Nuevo submódulo Python de la plataforma, hermano de `iot-server-background`,
`iot-server-infra` y `iot-server-web-app`. Contiene el runtime de **aprendizaje
federado con Flower**. La contribución (agregación ponderada por telemetría de
hardware) se implementa como una estrategia intercambiable por configuración.

> El runtime FL es Python; **no** va dentro de `iot-server-background/modules/`
> (ese patrón es solo para los servicios Go). Es su propia imagen Docker.

## Objetivo del Sprint 1 (M0)

Cerrar una ronda **FedAvg** local end-to-end.

**Definition of Done:** `flwr run . local-sim` cierra 3 rondas, `central_acc` sube
ronda a ronda, y se genera `checkpoints/global_r{N}.tflite`.

En M0 **no** intervienen MQTT / Kafka / PostgreSQL: es simulación local pura. Esos
componentes entran en Sprint 2, cuando se conectan los SBCs reales.

## Estructura

```
iot-server-fl/
├── pyproject.toml            # uv + configuración de la Flower App y federaciones
├── Dockerfile                # imagen del runtime FL (Python + Flower + TF-CPU)
├── docker-compose.yml        # servicio de simulación (M0)
├── fl_edge/
│   ├── __init__.py
│   ├── task.py               # [paso 2] modelo Keras, datos, eval, export .tflite
│   ├── records.py            # [paso 3] ArrayRecord <-> ndarrays
│   ├── client_app.py         # [paso 3] ClientApp (entrenamiento local)
│   ├── server_app.py         # [paso 4] ServerApp (orquestación de rondas)
│   ├── strategy.py           # [paso 4] FedAvg + TelemetryWeighted (stub)
│   └── strategies/
│       ├── __init__.py
│       └── registry.py       # [paso 4] get_strategy(name, **cfg)
└── db/
    └── 002_fl_schema.sql     # [paso 5] esquema FL
```

## Pasos del Sprint 1

1. **Andamiaje + entorno + Docker** (hecho).
2. `task.py` — modelo, datos (Fashion-MNIST), eval centralizada, export `.tflite`.
3. `records.py` + `client_app.py` — cliente que entrena y reporta pesos + timestamps.
4. `server_app.py` + `strategy.py` + `strategies/registry.py` — cierra el DoD.
5. `db/002_fl_schema.sql` — esquema FL.

## Ejecución con Docker (recomendado, consistente con tu infra)

```bash
cd iot-server-fl
docker compose build          # construye la imagen (TF + Flower + Ray: tarda)
docker compose up             # ejecuta el command actual (validación de entorno)
```

A partir del paso 4, en `docker-compose.yml` se cambia el `command` por
`flwr run . local-sim --stream` para correr la simulación real.

## Ejecución en host (alternativa, con uv)

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.10–3.12 (recomendado 3.11).

```bash
cd iot-server-fl
uv sync
uv run flwr --version
uv run python -c "import fl_edge, tensorflow as tf; print('fl_edge OK', tf.__version__)"
```

## Validación del paso 1

Aún NO corras la simulación completa (faltan los componentes, pasos 3–4). Valida
solo que el entorno construye y Flower está disponible:

```bash
docker compose build && docker compose up      # debe imprimir la versión de Flower
```

> Se usa la API nueva de Flower (basada en mensajes: `ServerApp`/`ClientApp`,
> `configure_train`/`aggregate_train`). Algunas firmas dependen de la versión
> instalada, así que comparte la versión de Flower que reporte el contenedor para
> afinar los pasos 3 y 4.
