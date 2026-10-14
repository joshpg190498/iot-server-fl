"""Conversión entre los pesos del modelo (lista de np.ndarray) y el ArrayRecord
de Flower. Aislar esto en un módulo evita esparcir detalles de la API de Flower
por todo el código: si una firma cambia entre versiones, se corrige aquí.
"""

from __future__ import annotations

from typing import List

import numpy as np

# La ubicación de ArrayRecord varió entre versiones (flwr.app vs flwr.common).
try:
    from flwr.app import ArrayRecord
except ImportError:  # pragma: no cover
    from flwr.common import ArrayRecord


def ndarrays_to_arrayrecord(ndarrays: List[np.ndarray]) -> "ArrayRecord":
    """Pesos (lista de arrays de Keras) -> ArrayRecord de Flower."""
    return ArrayRecord.from_numpy_ndarrays(ndarrays)


def arrayrecord_to_ndarrays(record: "ArrayRecord") -> List[np.ndarray]:
    """ArrayRecord de Flower -> pesos (lista de arrays de Keras)."""
    return record.to_numpy_ndarrays()