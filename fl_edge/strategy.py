"""Estrategia de agregación consciente de la salud del hardware (contribución).

TelemetryWeightedFedAvg hereda FedAvg (Flower 1.36, Message API) y cambia SOLO el
peso de cada actualización en `aggregate_train`:

    FedAvg:     w_{t+1} = Σ_k (n_k / Σ n_j) · w_k
    Propuesta:  w_{t+1} = Σ_k (ω_k / Σ ω_j) · w_k,   ω_k = n_k · h_k^γ

donde h_k sale de la telemetría del dispositivo en su ventana de entrenamiento
(ver fl_edge/aggregation/health.py). Con γ = 0 el resultado es idéntico a FedAvg,
y además queda registrada la telemetría de cada nodo: por eso el baseline de los
experimentos se corre como `telemetry_*` con γ = 0 (misma matemática que FedAvg,
mismo registro).

El cliente debe enviar en su respuesta:
    MetricRecord 'metrics': num-examples, steps-done, steps-planned,
                            train_started_unix, train_ended_unix, ...
    ConfigRecord 'node-info': device-id   (identidad del SBC en la plataforma)

Robustez: si la BD no responde o un nodo no tiene telemetría, su componente de
hardware queda neutral; un fallo de telemetría o de registro NUNCA impide agregar.
"""

from __future__ import annotations

import time
from logging import INFO, WARNING
from typing import Any, Dict, Iterable, List, Optional

try:
    from flwr.serverapp.strategy import FedAvg
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
except ImportError:  # pragma: no cover
    from flwr.server.strategy import FedAvg
    from flwr.server.strategy.strategy_utils import aggregate_arrayrecords

try:
    from flwr.common.logger import log
except ImportError:  # pragma: no cover
    import logging

    def log(level, msg, *args):  # type: ignore[no-redef]
        logging.getLogger("flwr").log(level, msg, *args)

from fl_edge.aggregation.health import HealthConfig, NodeInput, compute_weights

WEIGHT_KEY = "fl-weight"  # clave temporal con ω_k (se quita tras agregar)
NODE_INFO_KEY = "node-info"


class TelemetryWeightedFedAvg(FedAvg):
    def __init__(
        self,
        *args,
        gamma: float = 0.0,
        variant: str = "combined",
        lam: float = 2.0,
        alpha: float = 0.5,
        health_overrides: Optional[Dict[str, Any]] = None,
        db_url: Optional[str] = None,
        model_name: str = "fashion-cnn",
        run_id: Optional[str] = None,
        strategy_name: str = "telemetry_weighted",
        telemetry_grace_s: float = 5.0,
        persist: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.health_cfg = HealthConfig(variant=variant, gamma=gamma, lam=lam, alpha=alpha,
                                       **(health_overrides or {}))
        self.db_url = db_url
        self.model_name = model_name
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.strategy_name = strategy_name
        self.telemetry_grace_s = telemetry_grace_s
        self.persist = persist
        self.last_round: List[Dict[str, Any]] = []  # para tests y diagnóstico

    @property
    def gamma(self) -> float:
        return self.health_cfg.gamma

    def summary(self) -> None:
        super().summary()
        c = self.health_cfg
        log(INFO, "\t├──> Telemetría: variante=%s γ=%s λ=%s α=%s pesos(c,f,m,θ)=(%s,%s,%s,%s)",
            c.variant, c.gamma, c.lam, c.alpha, c.w_cpu, c.w_freq, c.w_mem, c.w_temp)

    # ------------------------------------------------------------------ helpers
    def _connect(self):
        try:
            import psycopg

            from fl_edge.registry.models import db_url_from_env

            return psycopg.connect(self.db_url or db_url_from_env(), connect_timeout=5)
        except Exception as exc:  # noqa: BLE001
            log(WARNING, "[telemetry] sin conexión a la BD (%s): componente de hardware neutral", exc)
            return None

    @staticmethod
    def _node_row(msg, weighted_by_key: str) -> Dict[str, Any]:
        content = msg.content
        metrics = next(iter(content.metric_records.values()))
        info = content.config_records.get(NODE_INFO_KEY) if hasattr(content, "config_records") else None
        device = str(info.get("device-id", "")) if info is not None else ""

        def num(key: str) -> Optional[float]:
            v = metrics.get(key)
            return float(v) if v is not None else None

        return {
            "node_id": str(getattr(msg.metadata, "src_node_id", "") or ""),  # identificador opaco (texto)
            "device_label": device,
            "num_examples": float(metrics[weighted_by_key]),
            "steps_done": num("steps-done"),
            "steps_planned": num("steps-planned"),
            "t_start": num("train_started_unix"),
            "t_end": num("train_ended_unix"),
            "train_loss": num("train_loss"),
            "train_accuracy": num("train_accuracy"),
        }

    def _attach_telemetry(self, rows: List[Dict[str, Any]], conn) -> None:
        from fl_edge.aggregation.telemetry import window_signals

        for r in rows:
            if conn is None or not r["device_label"] or r["t_start"] is None or r["t_end"] is None:
                continue
            try:
                r.update(window_signals(conn, r["device_label"], r["t_start"], r["t_end"]))
            except Exception as exc:  # noqa: BLE001
                log(WARNING, "[telemetry] %s: no se pudo leer la ventana (%s)", r["device_label"], exc)
                conn.rollback()

    # --------------------------------------------------------------- agregación
    def aggregate_train(self, server_round: int, replies: Iterable[Any]):
        valid_replies, _ = self._check_and_log_replies(list(replies), is_train=True)
        if not valid_replies:
            return None, None

        rows = [self._node_row(m, self.weighted_by_key) for m in valid_replies]

        # 1) Telemetría de cada ventana (la última muestra puede tardar unos segundos)
        conn = self._connect()
        if conn is not None and self.telemetry_grace_s > 0:
            time.sleep(self.telemetry_grace_s)
        self._attach_telemetry(rows, conn)

        # 2) Salud y peso de cada nodo
        results = compute_weights(
            [NodeInput(num_examples=r["num_examples"], steps_done=r["steps_done"],
                       steps_planned=r["steps_planned"], cpu_psi=r.get("cpu_psi"),
                       load_per_core=r.get("load_per_core"), mem_psi=r.get("mem_psi"),
                       ram_pct=r.get("ram_pct"), freq_ratio=r.get("freq_ratio"),
                       temp_max=r.get("temp_max")) for r in rows],
            self.health_cfg,
        )
        for r, h in zip(rows, results):
            r.update(sig_work=h.work, sig_cpu=h.cpu, sig_freq=h.freq, sig_mem=h.mem,
                     sig_temp=h.temp, hw_stress=h.hw, stress=h.stress, health=h.health,
                     weight=h.weight, weight_share=h.weight_share, missing_hw=h.missing_hw)

        # 3) Promedio ponderado por ω_k con la MISMA función que usa FedAvg
        contents = [m.content for m in valid_replies]
        for content, h in zip(contents, results):
            next(iter(content.metric_records.values()))[WEIGHT_KEY] = float(h.weight)
        try:
            arrays = aggregate_arrayrecords(contents, WEIGHT_KEY)
        finally:
            for content in contents:
                del next(iter(content.metric_records.values()))[WEIGHT_KEY]
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        self._log_round(server_round, rows)
        self.last_round = rows

        # 4) Registro (nunca bloquea el entrenamiento)
        if self.persist and conn is not None:
            try:
                from fl_edge.aggregation.persistence import insert_updates, upsert_round

                c = self.health_cfg
                round_id = upsert_round(conn, model_name=self.model_name, run_id=self.run_id,
                                        server_round=server_round, strategy=self.strategy_name,
                                        variant=c.variant, gamma=c.gamma, lam=c.lam)
                insert_updates(conn, round_id, rows)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                log(WARNING, "[telemetry] no se pudo registrar la ronda %s (%s)", server_round, exc)
        if conn is not None:
            conn.close()
        return arrays, metrics

    def record_central(self, server_round: int, accuracy: float, loss: float) -> None:
        """Lo llama el ServerApp tras evaluar el modelo global de la ronda."""
        if not self.persist or server_round < 1:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            from fl_edge.aggregation.persistence import record_central

            record_central(conn, model_name=self.model_name, run_id=self.run_id,
                           server_round=server_round, accuracy=accuracy, loss=loss)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            log(WARNING, "[telemetry] no se pudo registrar la evaluación de la ronda %s (%s)", server_round, exc)
        finally:
            conn.close()

    @staticmethod
    def _log_round(server_round: int, rows: List[Dict[str, Any]]) -> None:
        def f(v, fmt="{:.2f}"):
            return "  -  " if v is None else fmt.format(v)

        log(INFO, "[telemetry] ronda %s: salud y peso por nodo", server_round)
        log(INFO, "[telemetry]   %-10s %6s %6s %6s %6s %6s | %5s %5s %6s", "device",
            "trabajo", "cpuPSI", "frec", "memPSI", "temp", "s_k", "h_k", "peso")
        for r in rows:
            log(INFO, "[telemetry]   %-10s %6s %6s %6s %6s %6s | %5s %5s %5.1f%%%s",
                (r["device_label"] or f"node{r['node_id']}")[:10],
                f(r.get("steps_done") and r.get("steps_planned") and r["steps_done"] / r["steps_planned"]),
                f(r.get("cpu_psi"), "{:.1f}"), f(r.get("freq_ratio")), f(r.get("mem_psi"), "{:.1f}"),
                f(r.get("temp_max"), "{:.0f}"), f(r["stress"]), f(r["health"]),
                100 * r["weight_share"], "  (sin telemetría)" if r["missing_hw"] else "")
