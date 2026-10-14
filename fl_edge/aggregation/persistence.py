"""Registro de cada ronda y de cada actualización en ROUND_FL / UPDATES_FL.

Es la tabla que alimenta el dashboard MLOps y el análisis del Capítulo 4
(h_k y ω_k por dispositivo y ronda). Requiere 004_fl_rounds.sql.
Las funciones lanzan excepción; quien las llama decide (la estrategia nunca
deja que un fallo de registro tumbe el entrenamiento).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _model_id(cur, model_name: str) -> int:
    cur.execute(
        """INSERT INTO MODEL (NAME) VALUES (%s)
           ON CONFLICT (NAME) DO UPDATE SET NAME = EXCLUDED.NAME RETURNING ID""",
        (model_name,),
    )
    return cur.fetchone()[0]


def upsert_round(conn, *, model_name: str, run_id: str, server_round: int, strategy: str,
                 variant: str, gamma: float, lam: float) -> int:
    with conn.cursor() as cur:
        id_model = _model_id(cur, model_name)
        cur.execute(
            """INSERT INTO ROUND_FL (ID_MODEL, RUN_ID, ROUND_NUMBER, STRATEGY, VARIANT, GAMMA, LAMBDA, STATUS)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 'aggregated')
               ON CONFLICT (ID_MODEL, RUN_ID, ROUND_NUMBER) DO UPDATE SET
                   STRATEGY = EXCLUDED.STRATEGY, VARIANT = EXCLUDED.VARIANT,
                   GAMMA = EXCLUDED.GAMMA, LAMBDA = EXCLUDED.LAMBDA, STATUS = 'aggregated'
               RETURNING ID""",
            (id_model, run_id, server_round, strategy, variant, gamma, lam),
        )
        return cur.fetchone()[0]


def insert_updates(conn, round_id: int, rows: Iterable[Dict[str, Any]]) -> None:
    cols = ["ID_ROUND_FL", "NODE_ID", "ID_DEVICE", "DEVICE_LABEL", "NUM_EXAMPLES",
            "TRAIN_STARTED_UTC", "TRAIN_ENDED_UTC", "STEPS_DONE", "STEPS_PLANNED",
            "TRAIN_LOSS", "TRAIN_ACCURACY", "CPU_PSI", "MEM_PSI", "FREQ_RATIO", "TEMP_MAX",
            "LOAD_PER_CORE", "RAM_PCT", "TELEMETRY_SAMPLES", "SIG_WORK", "SIG_CPU", "SIG_FREQ",
            "SIG_MEM", "SIG_TEMP", "HW_STRESS", "STRESS", "HEALTH_WEIGHT", "AGGREGATION_WEIGHT"]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[2:])
    sql = (f"INSERT INTO UPDATES_FL ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT (ID_ROUND_FL, NODE_ID) DO UPDATE SET {updates}")
    with conn.cursor() as cur:
        for r in rows:
            label = r.get("device_label") or None
            # ID_DEVICE es FK a DEVICES: solo si el dispositivo está registrado
            id_device = None
            if label:
                cur.execute("SELECT 1 FROM DEVICES WHERE ID_DEVICE = %s", (label,))
                id_device = label if cur.fetchone() else None
            cur.execute(sql, (round_id, r["node_id"], id_device, label, r.get("num_examples"),
                              r.get("window_start"), r.get("window_end"), r.get("steps_done"),
                              r.get("steps_planned"), r.get("train_loss"), r.get("train_accuracy"),
                              r.get("cpu_psi"), r.get("mem_psi"), r.get("freq_ratio"), r.get("temp_max"),
                              r.get("load_per_core"), r.get("ram_pct"), r.get("samples"),
                              r.get("sig_work"), r.get("sig_cpu"), r.get("sig_freq"), r.get("sig_mem"),
                              r.get("sig_temp"), r.get("hw_stress"), r.get("stress"),
                              r.get("health"), r.get("weight_share")))


def record_central(conn, *, model_name: str, run_id: str, server_round: int,
                   accuracy: float, loss: float) -> None:
    """Tras la evaluación centralizada de la ronda: métricas del modelo global."""
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE ROUND_FL r SET CENTRAL_ACCURACY = %s, CENTRAL_LOSS = %s,
                   STATUS = 'completed', END_DATETIME_UTC = (NOW() AT TIME ZONE 'UTC')
               FROM MODEL m
               WHERE r.ID_MODEL = m.ID AND m.NAME = %s AND r.RUN_ID = %s AND r.ROUND_NUMBER = %s""",
            (accuracy, loss, model_name, run_id, server_round),
        )
