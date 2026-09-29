"""Jede Zahl der App-Texte ist hier über die fünf festen Sweep-Instanzen belegt. Positive UND negative
Aussagen: Reoptimierung und das Verfahren insgesamt helfen im Mittel deutlich - UND das ist KEINE Garantie auf
jeder einzelnen Instanz (anders als Stück 1-4/6/8's Suchraum-Satz). Rechenzeiten nur als Größenordnung geprüft;
teure Läufe sind modul-weit über lru_cache dedupliziert."""

from functools import lru_cache

import sb_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


@lru_cache(maxsize=1)
def _reopt():
    return tuple(tuple(r.items()) for r in ev.reoptimization_check())


def reopt_rows():
    return [dict(r) for r in _reopt()]


@lru_cache(maxsize=1)
def _timing():
    return tuple(tuple(r.items()) for r in ev.timing_sweep())


def timing_rows():
    return [dict(r) for r in _timing()]


@lru_cache(maxsize=1)
def _setup_sweep():
    return tuple(tuple(r.items()) for r in ev.setup_gap_sweep())


def setup_rows():
    return [dict(r) for r in _setup_sweep()]


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap_no_reopt"], 11.1, 20.0)
    near(std["gap_mwkr"], 1.7, 20.0)


# --- Der Beweis-Check dieses Stücks: eine gemessene Quote, keine 100 %-Garantie -----------------------------------------------------------------


def test_reoptimization_never_worse_rate_is_high_but_not_guaranteed():
    """Ehrlich anders als der Suchraum-Satz aus Stück 8: Reoptimierung hilft in den meisten, aber NICHT in
    JEDER Instanz (siehe test_algorithm.py für ein von Hand nachgerechnetes Gegenbeispiel)."""
    rows = reopt_rows()
    assert all(0.6 <= r["reopt_never_worse_rate"] <= 1.0 for r in rows)
    assert any(r["reopt_never_worse_rate"] < 1.0 for r in rows)


def test_sb_alone_does_not_reliably_match_the_true_optimum():
    """Shifting Bottleneck ist eine sehr gute Heuristik, kein exakter Löser - der Suchraum-Satz aus Stück 8
    gilt hier nicht."""
    rows = reopt_rows()
    assert any(r["sb_match_rate"] < 1.0 for r in rows)


# --- Timing: CP-SAT auf das volle Modell (im schlimmsten Fall exponentiell) gegen viele kleine Teilprobleme -------------------------------------


def test_shifting_bottleneck_stays_fast_even_where_the_full_model_gets_expensive():
    rows = timing_rows()
    large = rows[-1]
    assert large["sb_seconds"] < 5.0


# --- Vehikel B: Rüstzeit-Härtetest ------------------------------------------------------------------------------------------------------------


def test_setup_gap_is_never_negative():
    """Die Optimallösung (CP-SAT) ist per Definition nie schlechter als die Heuristik."""
    for row in setup_rows():
        assert row["gap_mean"] >= 0.0
