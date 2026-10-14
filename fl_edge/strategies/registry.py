"""Registro de estrategias: seleccionar la agregación por configuración.

    strategy = get_strategy("fedavg", fraction_train=1.0, min_train_nodes=2)

Cambiar de agregación = cambiar el nombre en la config del run, sin tocar código.

Variantes de la propuesta (misma clase, distinta configuración):
    telemetry_weighted / telemetry_combined : efecto (trabajo) + causas (hardware)
    telemetry_work                          : solo déficit de trabajo
    telemetry_hw                            : solo telemetría de hardware
Con gamma = 0 cualquiera de ellas equivale a FedAvg (y registra la telemetría).
Baselines pendientes en este sprint: fedprox, fednova.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict

try:
    from flwr.serverapp.strategy import FedAvg, Strategy
except ImportError:  # pragma: no cover
    from flwr.server.strategy import FedAvg, Strategy

from fl_edge.strategy import TelemetryWeightedFedAvg

# Nombre de estrategia -> fábrica que construye la instancia con **cfg.
_REGISTRY: Dict[str, Callable[..., "Strategy"]] = {
    "fedavg": FedAvg,
    "telemetry_weighted": partial(TelemetryWeightedFedAvg, variant="combined", strategy_name="telemetry_weighted"),
    "telemetry_combined": partial(TelemetryWeightedFedAvg, variant="combined", strategy_name="telemetry_combined"),
    "telemetry_work": partial(TelemetryWeightedFedAvg, variant="work", strategy_name="telemetry_work"),
    "telemetry_hw": partial(TelemetryWeightedFedAvg, variant="hw", strategy_name="telemetry_hw"),
}


def available_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


def is_telemetry_strategy(name: str) -> bool:
    return name.lower().strip().startswith("telemetry_")


def get_strategy(name: str, **cfg) -> "Strategy":
    """Devuelve una instancia de la estrategia `name` construida con `cfg`."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Estrategia desconocida: '{name}'. Disponibles: {available_strategies()}"
        )
    return _REGISTRY[key](**cfg)
