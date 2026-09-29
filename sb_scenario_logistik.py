"""Vehikel B "Werkstatt/Logistik": dieselben Aufträge wie Vehikel A, zusätzlich eine Familie JE AUFTRAG (alle
Operationen eines Auftrags teilen sie) und eine feste Rüstzeit beim Familienwechsel JE MASCHINE - identisch zum
Generator aus `job-shop-demo`, hier eigener Code für dieses Repo. Für Shifting Bottleneck macht das jedes
Ein-Maschinen-Teilproblem selbst zu einem kleinen Reihenfolgeproblem (sequenzabhängige Rüstzeit), nicht nur zu
einer einfachen Intervall-Einplanung."""

from dataclasses import dataclass

import numpy as np

import sb_constants as C


@dataclass(frozen=True)
class LogistikInstance:
    n: int
    m: int
    routing: np.ndarray
    proc: np.ndarray
    family: np.ndarray        # (n,): Familie je AUFTRAG
    setup: np.ndarray         # (F, F) Rüstzeit-Matrix
    seed: int


def generate(n, m, seed, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME, p_min=C.P_MIN, p_max=C.P_MAX):
    rng = np.random.default_rng(seed)
    routing = np.array([rng.permutation(m) for _ in range(n)])
    proc = rng.integers(p_min, p_max + 1, size=(n, m)).astype(np.int64)
    family = rng.integers(0, n_families, size=n).astype(np.int64)
    setup = np.full((n_families, n_families), setup_time, dtype=np.int64)
    np.fill_diagonal(setup, 0)
    return LogistikInstance(n, m, routing, proc, family, setup, seed)
