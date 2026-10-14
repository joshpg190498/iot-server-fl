"""Telemetría de un nodo dentro de su ventana de entrenamiento [t_ini, t_fin].

Lee las tablas crudas que llena el agente Go (vía data-reception):
    PRESSURE_STALL, CPU_FREQUENCY, CPU_TEMPERATURE, LOAD_AVERAGE, RAM_USAGE.

PSI con contadores acumulados (µs): la presión EXACTA en la ventana es
    100 · (total(b) − total(a)) / (t(b) − t(a))
con a = última muestra ≤ t_ini y b = primera muestra ≥ t_fin (o, si aún no llegó,
la última ≤ t_fin). No depende del promedio móvil de 10 s del kernel.

Todas las marcas en BD son UTC sin zona (COLLECTED_AT_UTC).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# Margen para buscar la muestra que "abre" y "cierra" la ventana (≥ 2 periodos de muestreo)
BRACKET = timedelta(seconds=30)


def utc_naive(unix_ts: float) -> datetime:
    return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).replace(tzinfo=None)


def _psi_window(cur, column: str, device: str, t0: datetime, t1: datetime) -> Optional[float]:
    cur.execute(
        f"""SELECT COLLECTED_AT_UTC, {column} FROM PRESSURE_STALL
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s
            ORDER BY COLLECTED_AT_UTC DESC LIMIT 1""",
        (device, t0 - BRACKET, t0),
    )
    a = cur.fetchone()
    cur.execute(
        f"""SELECT COLLECTED_AT_UTC, {column} FROM PRESSURE_STALL
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s
            ORDER BY COLLECTED_AT_UTC ASC LIMIT 1""",
        (device, t1, t1 + BRACKET),
    )
    b = cur.fetchone()
    if b is None:  # la muestra posterior aún no llegó: usar la última dentro de la ventana
        cur.execute(
            f"""SELECT COLLECTED_AT_UTC, {column} FROM PRESSURE_STALL
                WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s
                ORDER BY COLLECTED_AT_UTC DESC LIMIT 1""",
            (device, t0, t1),
        )
        b = cur.fetchone()
    if a is None:  # sin muestra previa: usar la primera dentro de la ventana
        cur.execute(
            f"""SELECT COLLECTED_AT_UTC, {column} FROM PRESSURE_STALL
                WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s
                ORDER BY COLLECTED_AT_UTC ASC LIMIT 1""",
            (device, t0, t1),
        )
        a = cur.fetchone()
    if not a or not b or a[1] is None or b[1] is None:
        return None
    dt_us = (b[0] - a[0]).total_seconds() * 1e6
    d_total = float(b[1]) - float(a[1])
    if dt_us <= 0 or d_total < 0:  # una sola muestra o reinicio del contador
        return None
    return min(100.0, 100.0 * d_total / dt_us)


def _scalar(cur, sql: str, params: tuple) -> Optional[float]:
    cur.execute(sql, params)
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def window_signals(conn, device: str, t_start_unix: float, t_end_unix: float) -> Dict[str, Any]:
    """Señales crudas del dispositivo en su ventana. Las ausentes vuelven como None."""
    t0, t1 = utc_naive(t_start_unix), utc_naive(t_end_unix)
    out: Dict[str, Any] = {"window_start": t0, "window_end": t1}
    with conn.cursor() as cur:
        out["cpu_psi"] = _psi_window(cur, "CPU_SOME_TOTAL", device, t0, t1)
        out["mem_psi"] = _psi_window(cur, "MEM_SOME_TOTAL", device, t0, t1)
        if out["cpu_psi"] is None:  # respaldo: promedio de avg10 dentro de la ventana
            out["cpu_psi"] = _scalar(cur, """SELECT AVG(CPU_SOME_AVG10) FROM PRESSURE_STALL
                WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        if out["mem_psi"] is None:
            out["mem_psi"] = _scalar(cur, """SELECT AVG(MEM_SOME_AVG10) FROM PRESSURE_STALL
                WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        out["freq_ratio"] = _scalar(cur, """SELECT AVG(FREQ_RATIO) FROM CPU_FREQUENCY
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        out["temp_max"] = _scalar(cur, """SELECT MAX(TEMPERATURE) FROM CPU_TEMPERATURE
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        out["ram_pct"] = _scalar(cur, """SELECT AVG(USED_PERCENT_RAM) FROM RAM_USAGE
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        load1 = _scalar(cur, """SELECT AVG(LOAD_AVERAGE_1M) FROM LOAD_AVERAGE
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1))
        cores = _scalar(cur, "SELECT CPU_COUNT FROM MAIN_DEVICE_INFORMATION WHERE ID_DEVICE = %s",
                        (device,)) or 4.0
        out["load_per_core"] = load1 / cores if load1 is not None else None
        out["samples"] = int(_scalar(cur, """SELECT COUNT(*) FROM CPU_FREQUENCY
            WHERE ID_DEVICE = %s AND COLLECTED_AT_UTC BETWEEN %s AND %s""", (device, t0, t1)) or 0)
    return out
