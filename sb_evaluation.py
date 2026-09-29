"""Auswertung der Shifting-Bottleneck-Demo: Shifting Bottleneck (mit Reoptimierung) gegen sich selbst OHNE
Reoptimierung, gegen MWKR (die beste einfache Regel aus Stück 8), gegen CP-SAT als exakte Gegenprobe (nur kleine
n), und das Vehikel-B-Experiment (Rüstzeit-Härtetest, wie in Stück 8). Beide Vergleiche (gegen ohne-Reoptimierung
UND gegen MWKR) können auf einzelnen Instanzen negativ werden - siehe [[feedback_vehicle_toggle_must_drive_primary_metrics]]-
Familie von Befunden: ein Verfahren, das im Mittel deutlich hilft, ist deshalb noch keine bewiesene Garantie auf
jeder einzelnen Instanz (anders als Stück 1-4/6, ähnlich wie Stück 7/8)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import sb_algorithm as A
import sb_constants as C
import sb_scenario as S
import sb_scenario_logistik as SL


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES


@lru_cache(maxsize=512)
def instance(n, m, seed):
    return S.generate(n, m, seed)


@lru_cache(maxsize=512)
def logistik_instance(n, m, seed, n_families, setup_time):
    return SL.generate(n, m, seed, n_families=n_families, setup_time=setup_time)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    sb: object                  # Hauptregel: Shifting Bottleneck MIT Reoptimierung
    sb_no_reopt: object         # derselbe Ablauf OHNE Schritt 4 (Reoptimierung)
    mwkr: object                # beste einfache Regel aus Stück 8 - die Vergleichsbasis dieses Stücks
    optimal: object             # None, wenn n > EXACT_MAX_N
    optimal_proven: bool

    @property
    def gap_no_reopt(self):
        return _gap(self.sb_no_reopt.cmax, self.sb.cmax)

    @property
    def gap_mwkr(self):
        return _gap(self.mwkr.cmax, self.sb.cmax)

    @property
    def sb_matches_optimum(self):
        return self.optimal is not None and abs(self.sb.cmax - self.optimal.cmax) < 1e-6

    @property
    def sb_ratio_to_optimum(self):
        return None if self.optimal is None or self.optimal.cmax <= 1e-9 else self.sb.cmax / self.optimal.cmax


def _gap(value, baseline):
    if baseline <= 1e-9:
        return 0.0 if value <= 1e-9 else float(value)
    return 100.0 * (value - baseline) / baseline


def analyse(settings):
    """Wertet Shifting Bottleneck auf dem gewählten Vehikel aus. MWKR (Stück 8) bleibt vehikel-bewusst als
    Vergleichsbasis - genau wie bei jedem Vorgängerstück ist auch hier kein negativer Abstand ausgeschlossen (das
    Verfahren ist bewiesen NIE schlechter als ohne Reoptimierung, aber NICHT bewiesen optimal)."""
    if settings.vehicle == "logistik":
        inst = logistik_instance(settings.n, settings.m, settings.seed, settings.n_families, settings.setup_time)
        routing, proc, family, setup = inst.routing, inst.proc, inst.family, inst.setup
        optimal, proven = (A.solve_exact(routing, proc, family, setup, C.EXACT_TIME_LIMIT_SECONDS)
                            if settings.n <= C.EXACT_MAX_N else (None, False))
        sb = A.shifting_bottleneck(routing, proc, family, setup, reoptimize=True)
        sb_no_reopt = A.shifting_bottleneck(routing, proc, family, setup, reoptimize=False)
        mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc), family, setup)
    else:
        inst = instance(settings.n, settings.m, settings.seed)
        routing, proc = inst.routing, inst.proc
        optimal, proven = (A.solve_exact(routing, proc, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
                            if settings.n <= C.EXACT_MAX_N else (None, False))
        sb = A.shifting_bottleneck(routing, proc, reoptimize=True)
        sb_no_reopt = A.shifting_bottleneck(routing, proc, reoptimize=False)
        mwkr = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))

    return Analysis(settings, inst, sb, sb_no_reopt, mwkr, optimal, proven)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        a = analyse(replace(s0, seed=seed))
        rows.append({"gap_no_reopt": a.gap_no_reopt, "gap_mwkr": a.gap_mwkr})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 4, 6, 8, 12, 16, 20), "m": (2, 3, 4, 5, 6)}
SWEEP_LABELS = {"n": "Aufträge", "m": "Maschinen"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def reoptimization_check(ns=(3, 4, 5, 6, 8), m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS):
    """Der zentrale Experiment-Check dieses Stücks: wie OFT hilft die Reoptimierung (nie schlechter als ohne) -
    ehrlich gemessen, KEINE 100 %-Garantie (siehe sb_algorithm.shifting_bottleneck-Docstring: die Reoptimierung
    verändert die Teilprobleme aller noch offenen Maschinen, ist also kein rein lokaler Verbesserungsschritt).
    Zusätzlich, wie oft Shifting Bottleneck allein das CP-SAT-Optimum trifft - ebenfalls keine 100 %-Erwartung,
    das Verfahren ist eine sehr gute Heuristik, kein exakter Löser."""
    rows = []
    for n in ns:
        never_worse, sb_matches = 0, 0
        for seed in seeds:
            inst = instance(n, m, seed)
            r_reopt = A.shifting_bottleneck(inst.routing, inst.proc, reoptimize=True)
            r_no = A.shifting_bottleneck(inst.routing, inst.proc, reoptimize=False)
            if r_reopt.cmax <= r_no.cmax + 1e-6:
                never_worse += 1
            opt, proven = A.solve_exact(inst.routing, inst.proc, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
            if proven and abs(r_reopt.cmax - opt.cmax) < 1e-6:
                sb_matches += 1
        rows.append({"value": n, "reopt_never_worse_rate": never_worse / len(seeds), "sb_match_rate": sb_matches / len(seeds)})
    return rows


def timing_sweep(ns=(2, 3, 4, 5, 6, 7, 8), m=C.DEFAULT_M, seed=C.DEFAULT_SEED):
    """Gemessene Rechenzeit: CP-SAT auf das VOLLE Job-Shop-Modell (im schlimmsten Fall exponentiell) gegen
    Shifting Bottleneck (m Teilprobleme je Iteration, jedes Teilproblem selbst über CP-SAT, aber auf nur EINER
    Maschine - in der Praxis um Größenordnungen schneller als das volle Modell)."""
    rows = []
    for n in ns:
        inst = instance(n, m, seed)
        t0 = time.perf_counter()
        A.solve_exact(inst.routing, inst.proc, time_limit_seconds=C.EXACT_TIME_LIMIT_SECONDS)
        t_exact = time.perf_counter() - t0
        t0 = time.perf_counter()
        A.shifting_bottleneck(inst.routing, inst.proc, reoptimize=True)
        t_sb = time.perf_counter() - t0
        rows.append({"value": n, "exact_seconds": t_exact, "sb_seconds": t_sb})
    return rows


def setup_gap(n=6, m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: Shifting Bottleneck (MIT Reoptimierung, rüstzeitbewusst über das Teilproblem) gegen
    die echte Optimallösung MIT Rüstzeiten (CP-SAT, deshalb kleines n)."""
    gaps = []
    for seed in seeds:
        linst = SL.generate(n, m, seed, n_families=n_families, setup_time=setup_time)
        sb = A.shifting_bottleneck(linst.routing, linst.proc, linst.family, linst.setup, reoptimize=True)
        opt, proven = A.solve_exact(linst.routing, linst.proc, linst.family, linst.setup, C.EXACT_TIME_LIMIT_SECONDS)
        if proven:
            gaps.append(_gap(sb.cmax, opt.cmax))
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=6, m=C.DEFAULT_M, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, m=m, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
