"""Registro de métricas por ronda a CSV, para curvas de aprendizaje y análisis
de sobreajuste (brecha train vs test).

El CSV resultante (results/metrics_*.csv) alimenta la capa de análisis (notebook)
donde se grafican las curvas y se calculan las métricas de la memoria.
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import Optional


class RoundCSVLogger:
    FIELDS = [
        "timestamp_utc",
        "strategy",
        "gamma",
        "server_round",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "overfit_gap",  # train_accuracy - test_accuracy (señal de sobreajuste)
    ]

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(
        self,
        strategy: str,
        gamma: float,
        server_round: int,
        train_loss: float,
        train_accuracy: float,
        test_loss: float,
        test_accuracy: float,
        timestamp_utc: Optional[str] = None,
    ) -> None:
        ts = timestamp_utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
        gap = round(train_accuracy - test_accuracy, 4)
        row = [
            ts,
            strategy,
            gamma,
            server_round,
            round(train_loss, 5),
            round(train_accuracy, 5),
            round(test_loss, 5),
            round(test_accuracy, 5),
            gap,
        ]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)