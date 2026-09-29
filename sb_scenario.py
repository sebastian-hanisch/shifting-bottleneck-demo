"""Vehikel A "Neutral" der Shifting-Bottleneck-Demo: identisches Modell zu `job-shop-demo` (Stück 8) - n
Aufträge, m Maschinen, jeder Auftrag besucht jede Maschine genau einmal, aber in AUFTRAGSEIGENER Reihenfolge
(`routing[j]` ist eine Permutation der Maschinen). Shifting Bottleneck ist ein besseres VERFAHREN für dasselbe
Problem, kein neues Modell - deshalb derselbe Generator (eigener Code, dieselbe Formensprache)."""

from dataclasses import dataclass

import numpy as np

import sb_constants as C


@dataclass(frozen=True)
class Instance:
    n: int
    m: int
    routing: np.ndarray     # (n, m): routing[j, pos] = Maschine der pos-ten Operation von Auftrag j
    proc: np.ndarray        # (n, m): proc[j, pos] = Bearbeitungszeit dieser Operation
    seed: int


def generate(n, m, seed, p_min=C.P_MIN, p_max=C.P_MAX):
    rng = np.random.default_rng(seed)
    routing = np.array([rng.permutation(m) for _ in range(n)])
    proc = rng.integers(p_min, p_max + 1, size=(n, m)).astype(np.int64)
    return Instance(n, m, routing, proc, seed)
