"""Fórmula de salud del hardware y peso de agregación (Python puro, sin Flower ni BD).

Para cada nodo k en una ronda:

    EFECTO   e_k = 1 - min(1, pasos_hechos / pasos_planificados)      (déficit de trabajo)
    CAUSAS   c_k contención de CPU   (PSI cpu, % de tiempo esperando CPU)
             f_k pérdida de frecuencia = 1 - frec_actual / frec_max   (throttling)
             m_k presión de memoria  (PSI memory)
             θ_k temperatura         (precursor del throttling)

    H_k = Σ w_i·x_i / Σ w_i      sobre las señales de hardware DISPONIBLES
    s_k = α·e_k + (1-α)·H_k       (variante 'combined')
        = e_k                     (variante 'work': solo efecto)
        = H_k                     (variante 'hw':   solo causas)
    h_k = exp(-λ·s_k)             salud ∈ [e^-λ, 1]
    ω_k = n_k · h_k^γ             γ = 0  ⇒  FedAvg exacto

Cada señal se normaliza a [0, 1] con un rango de referencia fijo (no min-max entre
nodos: con 2–4 nodos exageraría diferencias mínimas). Los rangos se calibran en el
piloto con línea base sin carga y con carga máxima.

Señal ausente -> se excluye y se renormalizan los pesos de las restantes.
Sin ninguna señal de hardware -> se marca `missing_hw` y el bloque de causas se
excluye (en 'combined', s_k = e_k; en 'hw', s_k = 0, es decir, neutral).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

VARIANTS = ("combined", "work", "hw")


@dataclass
class HealthConfig:
    variant: str = "combined"
    gamma: float = 0.0
    lam: float = 2.0
    alpha: float = 0.5  # peso del EFECTO frente a las CAUSAS (combined)
    # Pesos de las causas (se renormalizan sobre las disponibles)
    w_cpu: float = 0.35
    w_freq: float = 0.35
    w_mem: float = 0.20
    w_temp: float = 0.10
    # Rangos de normalización [lo, hi] -> [0, 1]
    cpu_psi_range: Tuple[float, float] = (5.0, 60.0)    # % tiempo esperando CPU
    load_range: Tuple[float, float] = (1.0, 3.0)        # respaldo: load1 / núcleos
    mem_psi_range: Tuple[float, float] = (0.0, 20.0)    # % tiempo esperando memoria
    ram_range: Tuple[float, float] = (70.0, 95.0)       # respaldo: % RAM usada
    temp_range: Tuple[float, float] = (70.0, 85.0)      # °C (Pi 4 limita entre 80 y 85)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variante desconocida '{self.variant}'; use una de {VARIANTS}")
        if self.lam < 0 or self.gamma < 0 or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("se requiere lam >= 0, gamma >= 0 y 0 <= alpha <= 1")


@dataclass
class NodeInput:
    """Lo que el servidor sabe de un nodo en la ronda."""
    num_examples: float
    steps_done: Optional[float] = None
    steps_planned: Optional[float] = None
    # Telemetría cruda de la ventana (None = no disponible)
    cpu_psi: Optional[float] = None
    load_per_core: Optional[float] = None
    mem_psi: Optional[float] = None
    ram_pct: Optional[float] = None
    freq_ratio: Optional[float] = None
    temp_max: Optional[float] = None


@dataclass
class NodeHealth:
    work: float                      # e_k
    cpu: Optional[float]             # c_k
    freq: Optional[float]            # f_k
    mem: Optional[float]             # m_k
    temp: Optional[float]            # θ_k
    hw: float                        # H_k
    stress: float                    # s_k
    health: float                    # h_k
    weight: float                    # ω_k (sin normalizar)
    weight_share: float = 0.0        # ω_k / Σ ω_j
    missing_hw: bool = False
    sources: Dict[str, str] = field(default_factory=dict)  # qué señal se usó (psi/load, ...)


def normalize(x: float, lo: float, hi: float) -> float:
    """Lineal a [0, 1], saturando fuera del rango."""
    if hi <= lo:
        raise ValueError(f"rango inválido [{lo}, {hi}]")
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def work_deficit(done: Optional[float], planned: Optional[float]) -> float:
    if done is None or planned is None or planned <= 0:
        return 0.0  # sin información de trabajo: neutral
    return 1.0 - min(1.0, max(0.0, done / planned))


def node_health(node: NodeInput, cfg: HealthConfig) -> NodeHealth:
    sources: Dict[str, str] = {}

    e = work_deficit(node.steps_done, node.steps_planned)

    c = None
    if node.cpu_psi is not None:
        c, sources["cpu"] = normalize(node.cpu_psi, *cfg.cpu_psi_range), "psi"
    elif node.load_per_core is not None:
        c, sources["cpu"] = normalize(node.load_per_core, *cfg.load_range), "load"

    f = None
    if node.freq_ratio is not None:
        f, sources["freq"] = min(1.0, max(0.0, 1.0 - node.freq_ratio)), "cpufreq"

    m = None
    if node.mem_psi is not None:
        m, sources["mem"] = normalize(node.mem_psi, *cfg.mem_psi_range), "psi"
    elif node.ram_pct is not None:
        m, sources["mem"] = normalize(node.ram_pct, *cfg.ram_range), "ram"

    t = None
    if node.temp_max is not None:
        t, sources["temp"] = normalize(node.temp_max, *cfg.temp_range), "sensor"

    pairs = [(x, w) for x, w in ((c, cfg.w_cpu), (f, cfg.w_freq), (m, cfg.w_mem), (t, cfg.w_temp))
             if x is not None and w > 0]
    total_w = sum(w for _, w in pairs)
    hw = sum(x * w for x, w in pairs) / total_w if total_w > 0 else 0.0
    missing_hw = total_w == 0

    if cfg.variant == "work":
        s = e
    elif cfg.variant == "hw":
        s = hw
    elif missing_hw:
        # Misma regla que dentro de H: el bloque ausente se excluye y se renormaliza.
        # Si contara como 0, la falta de telemetría reduciría a la mitad la
        # penalización por déficit de trabajo.
        s = e
    else:
        s = cfg.alpha * e + (1.0 - cfg.alpha) * hw

    h = math.exp(-cfg.lam * s)
    w = float(node.num_examples) * (h ** cfg.gamma)
    return NodeHealth(e, c, f, m, t, hw, s, h, w, missing_hw=missing_hw, sources=sources)


def compute_weights(nodes: List[NodeInput], cfg: HealthConfig) -> List[NodeHealth]:
    """Salud y peso de cada nodo de la ronda; completa la participación normalizada."""
    out = [node_health(n, cfg) for n in nodes]
    total = sum(r.weight for r in out)
    for r in out:
        r.weight_share = r.weight / total if total > 0 else 0.0
    return out
