"""Tests de TelemetryWeightedFedAvg contra la API real de Flower (FedAvg, ArrayRecord).

    uv run pytest -v tests/test_strategy.py
    # con BD de prueba (001..004 aplicados): lee telemetría y registra la ronda
    PG_TEST_URL=postgresql://user:pass@host:5432/db uv run pytest -v tests/test_strategy.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from fl_edge.strategies.registry import get_strategy


def reply(value: float, n: int, device: str = "", node_id: int = 1, steps=(100, 100),
          t=(1_000.0, 1_060.0)) -> Message:
    msg = Mock(spec=Message)
    msg.content = RecordDict({
        "arrays": ArrayRecord([np.full((2, 2), value, dtype=np.float32), np.array([value], dtype=np.float32)]),
        "metrics": MetricRecord({"num-examples": n, "train_loss": 0.5, "train_accuracy": 0.8,
                                 "steps-done": steps[0], "steps-planned": steps[1],
                                 "train_started_unix": t[0], "train_ended_unix": t[1]}),
        "node-info": ConfigRecord({"device-id": device}),
    })
    msg.has_error.side_effect = lambda: False
    msg.has_content.side_effect = lambda: True
    msg.metadata = Mock(src_node_id=node_id)
    return msg


def first_array(arrays: ArrayRecord) -> np.ndarray:
    return next(iter(arrays.values())).numpy()


def make(name="telemetry_weighted", **kw):
    base = dict(fraction_evaluate=0.0, min_train_nodes=2, min_available_nodes=2,
                persist=False, telemetry_grace_s=0.0, db_url="postgresql://nadie@127.0.0.1:1/x")
    base.update(kw)
    return get_strategy(name, **base)


def test_gamma_zero_matches_fedavg_exactly():
    replies = lambda: [reply(1.0, 600, steps=(10, 100)), reply(5.0, 400)]  # noqa: E731
    ours, _ = make(gamma=0.0).aggregate_train(1, replies())
    ref, _ = FedAvg(fraction_evaluate=0.0).aggregate_train(1, replies())
    np.testing.assert_allclose(first_array(ours), first_array(ref))
    np.testing.assert_allclose(first_array(ours), 1.0 * 0.6 + 5.0 * 0.4)


def test_work_deficit_down_weights_slow_node_without_db():
    # Sin BD (conexión imposible): solo el efecto (pasos) actúa; hardware neutral.
    s = make(gamma=1.0)
    arrays, metrics = s.aggregate_train(1, [reply(0.0, 500, "", 1, steps=(25, 100)),
                                            reply(10.0, 500, "", 2, steps=(100, 100))])
    slow, fast = s.last_round
    assert slow["sig_work"] == pytest.approx(0.75) and slow["missing_hw"]
    assert slow["weight_share"] < 0.5 < fast["weight_share"]
    # el promedio se corre hacia el nodo que completó su trabajo
    assert first_array(arrays)[0, 0] == pytest.approx(10.0 * fast["weight_share"], rel=1e-5)
    # la clave temporal de peso no queda en los replies ni en las métricas agregadas
    assert "fl-weight" not in metrics


def test_hw_variant_ignores_work():
    s = make("telemetry_hw", gamma=2.0)
    s.aggregate_train(1, [reply(0.0, 500, steps=(10, 100)), reply(1.0, 500)])
    assert all(r["health"] == 1.0 for r in s.last_round)  # sin telemetría y sin efecto -> neutral


# --------------------------------------------------------------------------- con BD
PG = os.getenv("PG_TEST_URL")
pg = pytest.mark.skipif(not PG, reason="PG_TEST_URL no definido")


def _seed_telemetry(conn, device: str, t0: datetime, seconds: int, cpu_psi_pct: float, freq: float,
                    temp: float):
    """Muestras cada 5 s, como el perfil experimental; PSI acumulado coherente con cpu_psi_pct."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO DEVICES (ID_DEVICE, DESCRIPTION) VALUES (%s, 'test') ON CONFLICT DO NOTHING",
                    (device,))
        for table in ("PRESSURE_STALL", "CPU_FREQUENCY", "CPU_TEMPERATURE"):
            cur.execute(f"DELETE FROM {table} WHERE ID_DEVICE = %s", (device,))
        total = 1_000_000
        for i in range(-2, seconds // 5 + 3):
            ts = t0 + timedelta(seconds=5 * i)
            total += int(cpu_psi_pct / 100 * 5e6)
            cur.execute("""INSERT INTO PRESSURE_STALL (ID_DEVICE, CPU_SOME_AVG10, CPU_SOME_AVG60, CPU_SOME_TOTAL,
                           MEM_SOME_AVG10, MEM_FULL_AVG10, MEM_SOME_TOTAL, MEM_FULL_TOTAL, COLLECTED_AT_UTC)
                           VALUES (%s,%s,%s,%s,0,0,%s,0,%s)""", (device, cpu_psi_pct, cpu_psi_pct, total, 500, ts))
            cur.execute("""INSERT INTO CPU_FREQUENCY (ID_DEVICE, CUR_FREQ_MHZ, MAX_FREQ_MHZ, FREQ_RATIO, COLLECTED_AT_UTC)
                           VALUES (%s, %s, 1800, %s, %s)""", (device, 1800 * freq, freq, ts))
            cur.execute("""INSERT INTO CPU_TEMPERATURE (ID_DEVICE, SENSOR_KEY, TEMPERATURE, COLLECTED_AT_UTC)
                           VALUES (%s, 'cpu_thermal', %s, %s)""", (device, temp, ts))
    conn.commit()


@pg
def test_end_to_end_with_real_telemetry_and_persistence():
    import psycopg

    t0 = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=5)
    start, end = t0.replace(tzinfo=timezone.utc).timestamp(), (t0 + timedelta(seconds=60)).replace(
        tzinfo=timezone.utc).timestamp()
    with psycopg.connect(PG) as conn:
        _seed_telemetry(conn, "TEST-HEALTHY", t0, 60, cpu_psi_pct=0.0, freq=1.0, temp=55)
        _seed_telemetry(conn, "TEST-STRESSED", t0, 60, cpu_psi_pct=90.0, freq=1.0, temp=62)

    run_id = f"pytest-{time.time():.0f}"
    s = make(gamma=1.0, db_url=PG, persist=True, run_id=run_id, model_name="itest-strategy")
    arrays, _ = s.aggregate_train(1, [
        reply(0.0, 500, "TEST-HEALTHY", 11, steps=(100, 100), t=(start, end)),
        reply(1.0, 500, "TEST-STRESSED", 22, steps=(40, 100), t=(start, end)),
    ])
    healthy, stressed = s.last_round
    # PSI exacto por contadores acumulados ≈ 90 %
    assert stressed["cpu_psi"] == pytest.approx(90.0, abs=0.5)
    assert healthy["cpu_psi"] == pytest.approx(0.0, abs=0.5)
    assert stressed["samples"] >= 12 and stressed["temp_max"] == 62
    assert stressed["weight_share"] < healthy["weight_share"]
    s.record_central(1, 0.87, 0.35)

    with psycopg.connect(PG) as conn:
        rows = conn.execute("""
            SELECT u.DEVICE_LABEL, u.ID_DEVICE, u.STEPS_DONE, round(u.CPU_PSI::numeric, 1),
                   u.HEALTH_WEIGHT, u.AGGREGATION_WEIGHT, r.CENTRAL_ACCURACY, r.STATUS
            FROM UPDATES_FL u JOIN ROUND_FL r ON r.ID = u.ID_ROUND_FL
            WHERE r.RUN_ID = %s ORDER BY u.NODE_ID""", (run_id,)).fetchall()
        conn.execute("DELETE FROM MODEL WHERE NAME = 'itest-strategy'")
    assert [r[0] for r in rows] == ["TEST-HEALTHY", "TEST-STRESSED"]
    assert rows[1][1] == "TEST-STRESSED" and rows[1][2] == 40
    assert rows[1][4] < rows[0][4] == 1.0
    assert sum(r[5] for r in rows) == pytest.approx(1.0)
    assert rows[0][6] == pytest.approx(0.87) and rows[0][7] == "completed"
