"""fl_edge.client_app — ClientApp de Flower (corre en cada SuperNode / SBC).

Recibe el modelo global, entrena con los datos locales y devuelve los pesos
actualizados junto con `num-examples` y los timestamps de inicio/fin del
entrenamiento. Esos timestamps son la semilla de la telemetría: en la
contribución (estrategia por telemetría) permiten cruzar la ventana de
entrenamiento de cada nodo con sus métricas de hardware.
"""

from __future__ import annotations

import time

# --- Imports defensivos (la organización de módulos varió entre versiones) ---
try:
    from flwr.clientapp import ClientApp
except ImportError:  # pragma: no cover
    from flwr.client import ClientApp

try:
    from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict
except ImportError:  # pragma: no cover
    from flwr.common import ConfigRecord, Context, Message, MetricRecord, RecordDict

from fl_edge.preflight import check_cpu
from fl_edge.records import arrayrecord_to_ndarrays, ndarrays_to_arrayrecord
from fl_edge.task import build_model, get_weights, load_partition, set_weights, train_local

# Falla claro si el CPU no cumple requisitos (p.ej. sin AVX en x86).
check_cpu()

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Config del nodo (la asigna el motor de simulación o el despliegue real)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    # Identidad del SBC en la plataforma (ID_DEVICE). Permite al servidor cruzar la
    # ventana de entrenamiento con la telemetría. Vacío = nodo sin telemetría (neutral).
    device_id = str(context.node_config.get("device-id", ""))

    # Config del run (definida en pyproject [tool.flwr.app.config])
    dataset = str(context.run_config["dataset"])
    local_epochs = int(context.run_config["local-epochs"])
    batch_size = int(context.run_config["batch-size"])
    lr = float(context.run_config["learning-rate"])
    # Presupuesto de tiempo del entrenamiento local (0 = sin presupuesto)
    time_budget_s = float(context.run_config.get("local-time-budget-s", 0.0))

    # Datos locales de este nodo
    data = load_partition(partition_id, num_partitions, dataset=dataset)

    # Modelo con los pesos globales entrantes
    model = build_model(dataset, lr=lr)
    global_ndarrays = arrayrecord_to_ndarrays(msg.content["arrays"])
    set_weights(model, global_ndarrays)

    # Entrenamiento local, capturando la ventana temporal (para telemetría)
    t_start = time.time()
    n_examples, metrics = train_local(
        model, data, epochs=local_epochs, batch_size=batch_size, lr=lr,
        time_budget_s=time_budget_s,
    )
    t_end = time.time()

    # Respuesta: pesos + métricas (trabajo y ventana) + identidad del nodo.
    # Todas las respuestas llevan las MISMAS claves (Flower lo valida al agregar).
    reply = RecordDict(
        {
            "arrays": ndarrays_to_arrayrecord(get_weights(model)),
            "metrics": MetricRecord(
                {
                    "num-examples": n_examples,
                    "train_loss": metrics["train_loss"],
                    "train_accuracy": metrics["train_accuracy"],
                    "steps-done": int(metrics["steps_done"]),
                    "steps-planned": int(metrics["steps_planned"]),
                    "train_started_unix": t_start,
                    "train_ended_unix": t_end,
                }
            ),
            "node-info": ConfigRecord({"device-id": device_id}),
        }
    )
    return Message(content=reply, reply_to=msg)