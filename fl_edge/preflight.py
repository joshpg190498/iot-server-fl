"""Chequeos de entorno previos a importar frameworks pesados (TensorFlow).

Convierte fallos crípticos —p.ej. 'Illegal instruction' / EXIT=132 por falta de
AVX— en mensajes claros y accionables, sin importar dónde corra el contenedor.
"""

from __future__ import annotations

import platform


def has_avx() -> bool:
    """True si el CPU expone AVX. Si no se puede determinar, asume True (no bloquea)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            text = f.read()
    except OSError:
        return True
    for line in text.splitlines():
        # Linux x86: 'flags'. Algunos entornos ARM: 'Features'.
        if line.startswith("flags") or line.startswith("Features"):
            return "avx" in line.split()
    return True


def check_cpu() -> None:
    """Valida requisitos de CPU. Solo aplica a x86_64 (las ruedas de TF piden AVX).

    En aarch64 (Raspberry Pi) TF usa ruedas ARM que no requieren AVX: no se valida.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return

    if not has_avx():
        raise SystemExit(
            "\n[preflight] ERROR: esta CPU no expone AVX.\n"
            "Las ruedas oficiales de TensorFlow (x86_64) requieren AVX; sin él el\n"
            "proceso muere con 'Illegal instruction' (EXIT=132).\n\n"
            "Cómo resolverlo:\n"
            "  - Compute Engine / servidores modernos: ya tienen AVX/AVX2 (sin accion).\n"
            "  - VirtualBox: habilita AVX en la VM\n"
            "      VBoxManage setextradata \"<VM>\" VBoxInternal/CPUM/IsaExts/AVX  1\n"
            "      VBoxManage setextradata \"<VM>\" VBoxInternal/CPUM/IsaExts/AVX2 1\n"
            "    o corre en el host / instancia cloud. Ver README.\n"
        )