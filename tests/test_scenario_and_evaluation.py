"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweep, Reoptimierungs-Check, Timing-Messreihe, Vehikel-B-Härtetest (Rüstzeiten)."""

from dataclasses import replace

import numpy as np
import pytest

import sb_algorithm as A
import sb_constants as C
import sb_evaluation as ev
import sb_scenario as S
import sb_scenario_logistik as SL


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 4, 3)
    assert inst.n == 20 and inst.m == 4
    assert inst.routing.shape == (20, 4) and inst.proc.shape == (20, 4)
    assert inst.proc.min() >= C.P_MIN and inst.proc.max() <= C.P_MAX
    for j in range(20):
        assert sorted(inst.routing[j].tolist()) == list(range(4))        # jede Maschine genau einmal


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(15, 3, 5), S.generate(15, 3, 5), S.generate(15, 3, 6)
    assert np.array_equal(a.routing, b.routing) and np.array_equal(a.proc, b.proc)
    assert not np.array_equal(a.proc, c.proc)


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shares_the_same_routing_and_processing_times_as_neutral():
    neutral = S.generate(15, 3, 7)
    logistik = SL.generate(15, 3, 7)
    assert np.array_equal(neutral.routing, logistik.routing) and np.array_equal(neutral.proc, logistik.proc)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 3, 2), SL.generate(10, 3, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup)


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=10, m=4))
    assert a.optimal is None


def test_analysis_respects_small_n_cp_sat_gegenprobe():
    a = ev.analyse(ev.Settings(n=4, m=3))
    assert a.optimal is not None and a.optimal_proven


# --- Vehikel-Bewusstsein der Hauptanalyse (von Anfang an, siehe [[feedback_vehicle_toggle_must_drive_primary_metrics]]) ----------------------


def test_analyse_on_the_logistik_vehicle_actually_uses_setup_aware_completion_times():
    settings = ev.Settings(n=8, m=4, seed=100000, vehicle="logistik", setup_time=30, n_families=3)
    a = ev.analyse(settings)
    linst = ev.logistik_instance(8, 4, 100000, 3, 30)
    independent = A.shifting_bottleneck(linst.routing, linst.proc, linst.family, linst.setup, reoptimize=True)
    assert a.sb.cmax == pytest.approx(independent.cmax)


def test_analyse_on_the_neutral_vehicle_is_unaffected_by_logistik_only_settings():
    a1 = ev.analyse(ev.Settings(n=10, m=4, seed=5, vehicle="neutral", setup_time=5))
    a2 = ev.analyse(ev.Settings(n=10, m=4, seed=5, vehicle="neutral", setup_time=60))
    assert a1.sb.cmax == pytest.approx(a2.sb.cmax)


def test_switching_vehicle_actually_changes_the_sb_cmax():
    a_neutral = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="neutral"))
    a_logistik = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="logistik", setup_time=60, n_families=2))
    assert a_neutral.sb.cmax != pytest.approx(a_logistik.sb.cmax)


def test_sb_at_zero_setup_time_matches_the_neutral_vehicle_exactly():
    a1 = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="neutral"))
    a2 = ev.analyse(ev.Settings(n=10, m=4, seed=7, vehicle="logistik", setup_time=0))
    assert a1.sb.cmax == pytest.approx(a2.sb.cmax)


def test_gap_no_reopt_can_be_negative_even_on_the_neutral_vehicle():
    """Der ehrliche Befund dieses Stücks (siehe test_algorithm.py): bei n=3, m=4, Seed 100000 schneidet die
    Reoptimierung minimal schlechter ab als ganz ohne - der Beweis-Check gilt für den MITTELWERT über eine
    Messreihe, nicht für jede einzelne Instanz."""
    a = ev.analyse(ev.Settings(n=3, m=4, seed=100000))
    assert a.gap_no_reopt < 0.0


def test_gap_mwkr_can_also_be_negative():
    """n=10, m=4, Seed 100000: Shifting Bottleneck schneidet hier schlechter ab als MWKR (Stück 8) - auch das
    Verfahren insgesamt ist keine bewiesene Garantie gegenüber jeder einzelnen Regel auf jeder Instanz."""
    a = ev.analyse(ev.Settings(n=10, m=4, seed=100000))
    assert a.gap_mwkr < 0.0


def test_analysis_is_deterministic():
    s = ev.Settings(n=10, m=4, seed=1)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.sb.cmax == pytest.approx(b.sb.cmax)
    assert a.mwkr.cmax == pytest.approx(b.mwkr.cmax)


# --- Sweep und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=10, m=4))
    assert r["n_runs"] == len(C.SWEEP_SEEDS)


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 15))
    assert [r["value"] for r in rows] == [5, 15]
    rows_m = ev.sweep("m", ev.Settings(), (2, 4))
    assert [r["value"] for r in rows_m] == [2, 4]


def test_reoptimization_check_reports_both_rates_and_neither_is_a_forced_constant():
    rows = ev.reoptimization_check(ns=(3, 4, 5), seeds=C.SWEEP_SEEDS)
    assert all(0.0 <= r["reopt_never_worse_rate"] <= 1.0 for r in rows)
    assert all(0.0 <= r["sb_match_rate"] <= 1.0 for r in rows)
    assert any(r["reopt_never_worse_rate"] < 1.0 for r in rows)   # der ehrliche Befund, nicht immer 100 %


def test_timing_sweep_shows_sb_growing_far_slower_than_exact_in_the_worst_case():
    rows = ev.timing_sweep(ns=(2, 8))
    small, large = rows[0], rows[1]
    assert large["sb_seconds"] < 5.0


def test_setup_gap_is_never_negative():
    """Shifting Bottleneck ist eine Heuristik, die Optimallösung per Definition nie schlechter."""
    row = ev.setup_gap(n=6, setup_time=15)
    assert row["gap_mean"] >= 0.0
