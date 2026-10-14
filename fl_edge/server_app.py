"""fl_edge.server_app — ServerApp de Flower (orquesta las rondas en el servidor).

Inicializa el modelo global, ejecuta las rondas FedAvg, evalúa de forma
centralizada tras cada ronda (sobre test Y sobre un subconjunto de train, para
poder graficar curvas de aprendizaje y detectar sobreajuste), exporta el modelo
global a .tflite por ronda y registra las métricas en CSV.

[registro] Al terminar el entrenamiento, el modelo global FINAL se registra
como una nueva versión en MODEL / MODEL_VERSION (con sus métricas) y se copia a
artifacts/, quedando listo para desplegar en los SBC. Un fallo del registro
nunca aborta el entrenamiento. Config opcional en [tool.flwr.app.config]:
    model-name     = "fashion-cnn"   (nombre del modelo en el registro)
    register-model = true            (false para simulaciones sin BD)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fl_edge.preflight import check_cpu

# Falla claro si el CPU no cumple requisitos (p.ej. sin AVX en x86).
check_cpu()

# --- Imports defensivos (organización de módulos según versión de Flower) ---
try:
    from flwr.app import ConfigRecord, Context, MetricRecord
except ImportError:  # pragma: no cover
    from flwr.common import ConfigRecord, Context, MetricRecord

try:
    from flwr.serverapp import Grid, ServerApp
except ImportError:  # pragma: no cover
    from flwr.server import Grid, ServerApp

from fl_edge.metrics import RoundCSVLogger
from fl_edge.records import arrayrecord_to_ndarrays, ndarrays_to_arrayrecord
from fl_edge.registry.models import try_register_version  # [registro]
from fl_edge.strategies.registry import get_strategy, is_telemetry_strategy
from fl_edge.task import (
    build_model,
    evaluate,
    export_tflite,
    load_centralized_testset,
    load_centralized_trainset,
    save_keras,
    set_weights,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run = context.run_config
    num_rounds = int(run["num-server-rounds"])
    fraction_train = float(run["fraction-train"])
    min_nodes = int(run["min-train-nodes"])
    timeout = float(run["round-timeout-seconds"])
    dataset = str(run["dataset"])
    strategy_name = str(run["strategy"])
    gamma = float(run["gamma"])
    lr = float(run["learning-rate"])
    # [registro] opcionales: si no están en pyproject se usan estos valores
    model_name = str(run.get("model-name", "fashion-cnn"))
    register_model = _as_bool(run.get("register-model", True)) and _as_bool(
        os.getenv("FL_REGISTER_MODEL", "true")
    )

    # [registro] estado de la última ronda (para registrar el modelo final)
    last = {"round": 0, "tflite": None, "keras": None, "metrics": {}}

    # 1) Modelo global inicial -> ArrayRecord inicial
    model = build_model(dataset, lr=lr)
    initial_arrays = ndarrays_to_arrayrecord(model.get_weights())

    # 2) Conjuntos para evaluacion centralizada por ronda
    testset = load_centralized_testset(dataset)
    trainset_eval = load_centralized_trainset(dataset, max_samples=5000)

    # 3) Logger de metricas por ronda (curvas de aprendizaje + sobreajuste)
    run_label = time.strftime("%Y%m%d-%H%M%S")
    csv_path = f"results/metrics_{strategy_name}_{run_label}.csv"
    metrics_logger = RoundCSVLogger(csv_path)

    # 4) Evaluacion centralizada: tras cada ronda, sobre train y test.
    def global_evaluate(server_round: int, arrays) -> "MetricRecord":
        set_weights(model, arrayrecord_to_ndarrays(arrays))
        test_loss, test_acc = evaluate(model, testset)
        train_loss, train_acc = evaluate(model, trainset_eval)
        gap = train_acc - test_acc

        metrics_logger.log(
            strategy=strategy_name,
            gamma=gamma,
            server_round=server_round,
            train_loss=train_loss,
            train_accuracy=train_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
        )

        # Export del modelo global de la ronda (.tflite + checkpoint Keras).
        # [registro] se guarda la ruta de la última ronda; ronda 0 = modelo inicial
        last["round"] = server_round
        last["tflite"] = last["keras"] = None
        last["metrics"] = {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
        }
        try:
            path = export_tflite(model, server_round)
            keras_path = save_keras(model, server_round)
            last["tflite"] = _existing_path(path)
            last["keras"] = _existing_path(keras_path)
            export_note = f"-> {path}"
        except Exception as exc:  # noqa: BLE001
            export_note = f"(export fallo: {exc})"

        # [telemetría] precisión del modelo global de la ronda -> ROUND_FL
        if hasattr(strategy, "record_central"):
            strategy.record_central(server_round, float(test_acc), float(test_loss))

        print(
            f"[server] ronda {server_round}  test_acc={test_acc:.4f}  "
            f"train_acc={train_acc:.4f}  sobreajuste(gap)={gap:+.4f}  "
            f"test_loss={test_loss:.4f}  {export_note}"
        )
        return MetricRecord({"central_loss": test_loss, "central_accuracy": test_acc})

    # 5) Estrategia via registry
    strategy_cfg = dict(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,        # M0/M1: solo evaluacion centralizada
        min_train_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )
    if is_telemetry_strategy(strategy_name):
        # [telemetría] parámetros de la fórmula (opcionales en pyproject)
        strategy_cfg.update(
            gamma=gamma,
            lam=float(run.get("lambda", 2.0)),
            alpha=float(run.get("alpha-work", 0.5)),
            model_name=model_name,
            run_id=f"{strategy_name}-{run_label}",
            telemetry_grace_s=float(run.get("telemetry-grace-s", 5.0)),
            persist=_as_bool(run.get("persist-rounds", True)),
        )
    strategy = get_strategy(strategy_name, **strategy_cfg)

    # 6) Config de entrenamiento enviada a los clientes
    train_config = ConfigRecord(
        {
            "local-epochs": int(run["local-epochs"]),
            "batch-size": int(run["batch-size"]),
            "learning-rate": lr,
        }
    )

    print(
        f"[server] estrategia={strategy_name} | rondas={num_rounds} | "
        f"min_nodes={min_nodes} | dataset={dataset} | gamma={gamma}"
    )
    print(f"[server] metricas por ronda -> {csv_path}")

    # 7) Ejecutar el aprendizaje federado
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=timeout,
        train_config=train_config,
        evaluate_fn=global_evaluate,
    )

    print("[server] entrenamiento federado completo")
    print(f"[server] curvas de aprendizaje guardadas en {csv_path}")

    # 8) [registro] Registrar el modelo global final como nueva versión
    if not register_model:
        print("[registry] registro desactivado (register-model=false); no se versiona el modelo")
        return
    if last["round"] < 1:
        print("[registry] AVISO: no se completó ninguna ronda; no se registra el modelo")
        return
    try:
        num_params = int(model.count_params())
    except Exception:  # noqa: BLE001
        num_params = None
    try_register_version(
        model_name=model_name,
        tflite_path=last["tflite"],
        keras_path=last["keras"],
        num_params=num_params,
        metrics={
            **last["metrics"],
            "strategy": strategy_name,
            "gamma": gamma,
            "num_rounds": num_rounds,
            "rounds_completed": last["round"],
            "dataset": dataset,
            "local_epochs": int(run["local-epochs"]),
            "learning_rate": lr,
            "metrics_csv": csv_path,
        },
    )


def _as_bool(value) -> bool:
    """Acepta true/false de pyproject (bool) o de variables de entorno (str)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _existing_path(value) -> str | None:
    """Ruta absoluta si `value` es un archivo existente; si no, None."""
    if not value:
        return None
    p = Path(str(value))
    return str(p.resolve()) if p.is_file() else None