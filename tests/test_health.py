"""Tests de la fórmula de salud (Python puro).

    uv run pytest -v tests/test_health.py
"""

import math

import pytest

from fl_edge.aggregation.health import HealthConfig, NodeInput, compute_weights, node_health, normalize, work_deficit
from fl_edge.task import planned_steps


def test_normalize_saturates():
    assert normalize(70, 70, 85) == 0.0
    assert normalize(77.5, 70, 85) == 0.5
    assert normalize(90, 70, 85) == 1.0
    assert normalize(10, 70, 85) == 0.0


def test_work_deficit():
    assert work_deficit(100, 100) == 0.0
    assert work_deficit(25, 100) == 0.75
    assert work_deficit(150, 100) == 0.0      # más pasos de lo planificado no es "negativo"
    assert work_deficit(None, 100) == 0.0     # sin información: neutral


def test_gamma_zero_is_fedavg():
    cfg = HealthConfig(gamma=0.0)
    res = compute_weights([NodeInput(600, 10, 100, cpu_psi=90, freq_ratio=0.5),
                           NodeInput(400, 100, 100, cpu_psi=0, freq_ratio=1.0)], cfg)
    assert [r.weight for r in res] == [600.0, 400.0]
    assert res[0].weight_share == pytest.approx(0.6)


def test_stressed_node_gets_less_weight():
    cfg = HealthConfig(gamma=1.0)
    healthy = NodeInput(500, 100, 100, cpu_psi=0.0, mem_psi=0.0, freq_ratio=1.0, temp_max=55)
    stressed = NodeInput(500, 40, 100, cpu_psi=92.0, mem_psi=0.0, freq_ratio=1.0, temp_max=60)
    h, s = compute_weights([healthy, stressed], cfg)
    assert h.stress == 0.0 and h.health == 1.0
    # e=0.6; c=1 (PSI 92% satura); f=0; m=0; θ=0 -> H=0.35 -> s=0.5·0.6+0.5·0.35=0.475
    assert s.work == pytest.approx(0.6)
    assert s.hw == pytest.approx(0.35)
    assert s.stress == pytest.approx(0.475)
    assert s.health == pytest.approx(math.exp(-2 * 0.475))
    assert h.weight_share > s.weight_share
    assert h.weight_share + s.weight_share == pytest.approx(1.0)


def test_variants_isolate_effect_and_causes():
    n = NodeInput(100, 50, 100, cpu_psi=60.0, freq_ratio=1.0)
    assert node_health(n, HealthConfig(variant="work")).stress == pytest.approx(0.5)
    hw = node_health(n, HealthConfig(variant="hw"))
    assert hw.stress == pytest.approx(hw.hw) and hw.stress > 0


def test_missing_telemetry_is_neutral_not_penalized():
    r = node_health(NodeInput(100, 100, 100), HealthConfig(gamma=2.0))
    assert r.missing_hw and r.hw == 0.0 and r.health == 1.0 and r.weight == 100.0


def test_fallbacks_when_psi_missing():
    r = node_health(NodeInput(100, 100, 100, load_per_core=3.0, ram_pct=95.0), HealthConfig())
    assert r.sources == {"cpu": "load", "mem": "ram"}
    assert r.cpu == 1.0 and r.mem == 1.0


def test_available_weights_are_renormalized():
    # Solo frecuencia disponible: H = f, no 0.35·f
    r = node_health(NodeInput(100, freq_ratio=0.5), HealthConfig(variant="hw"))
    assert r.hw == pytest.approx(0.5)


def test_throttled_frequency_counts():
    r = node_health(NodeInput(100, freq_ratio=1 / 3), HealthConfig(variant="hw"))
    assert r.freq == pytest.approx(2 / 3)


def test_invalid_config():
    with pytest.raises(ValueError):
        HealthConfig(variant="otra")
    with pytest.raises(ValueError):
        HealthConfig(alpha=1.5)


def test_planned_steps():
    assert planned_steps(30000, 1, 32) == 938
    assert planned_steps(30000, 2, 32) == 1876
    assert planned_steps(0, 1, 32) == 0
