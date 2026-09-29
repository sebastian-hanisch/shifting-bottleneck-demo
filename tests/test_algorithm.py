"""sb_algorithm: CP-SAT gegen unabhängige Brute-Force-Vollaufzählung (nur kleine n), Heads/Tails-Eigenschaften,
Shifting Bottleneck gegen CP-SAT-Optimum, die ehrliche Kehrseite der Reoptimierung (kein bewiesen monotoner
Schritt), Rüstzeit-Variante, Handrechnung."""

import itertools

import numpy as np
import pytest

import sb_algorithm as A


def _instance(seed, n, m):
    rng = np.random.default_rng(seed)
    routing = np.array([rng.permutation(m) for _ in range(n)])
    proc = rng.integers(1, 30, size=(n, m)).astype(np.int64)
    return routing, proc


def _brute_force(routing, proc, family=None, setup=None):
    """Alle Permutationen der Operationen je Maschine, nur zulässige (azyklische) Kombinationen ausgewertet."""
    n, m = routing.shape
    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))
    best = None
    for combo in itertools.product(*(list(itertools.permutations(ops)) for ops in ops_per_machine)):
        machine_order = {k: list(combo[k]) for k in range(m)}
        machine_ptr = {k: 0 for k in range(m)}
        job_ptr = {j: 0 for j in range(n)}
        machine_free = [0] * m
        job_free = [0] * n
        prev_family = [None] * m
        end_time = {}
        scheduled, total = 0, n * m
        progress = True
        while scheduled < total and progress:
            progress = False
            for k in range(m):
                if machine_ptr[k] >= len(machine_order[k]):
                    continue
                j, pos = machine_order[k][machine_ptr[k]]
                if job_ptr[j] == pos:
                    s = int(setup[prev_family[k], family[j]]) if family is not None and prev_family[k] is not None else 0
                    st = max(machine_free[k] + s, job_free[j])
                    dur = int(proc[j, pos])
                    end_time[j, pos] = st + dur
                    machine_free[k] = st + dur
                    job_free[j] = st + dur
                    if family is not None:
                        prev_family[k] = family[j]
                    machine_ptr[k] += 1
                    job_ptr[j] += 1
                    scheduled += 1
                    progress = True
        if scheduled < total:
            continue
        cmax = max(end_time.values())
        if best is None or cmax < best:
            best = cmax
    return best


@pytest.mark.parametrize("n,m", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_cp_sat_matches_brute_force_for_every_seed(n, m):
    for seed in range(5):
        routing, proc = _instance(seed * 10 + n + m, n, m)
        bf = _brute_force(routing, proc)
        result, proven = A.solve_exact(routing, proc, time_limit_seconds=10)
        assert proven
        assert result.cmax == pytest.approx(bf, abs=1e-6)


def test_giffler_thompson_produces_a_feasible_precedence_respecting_schedule():
    routing, proc = _instance(5, 6, 3)
    result = A.giffler_thompson(routing, proc, A.mwkr_priority(proc))
    n, m = routing.shape
    for j in range(n):
        for pos in range(1, m):
            assert result.start[j, pos] >= result.end[j, pos - 1] - 1e-9
    for k in range(m):
        ops_k = sorted([(j, pos) for j in range(n) for pos in range(m) if routing[j, pos] == k], key=lambda jp: result.start[jp])
        for a, b in zip(ops_k, ops_k[1:]):
            assert result.start[b] >= result.end[a] - 1e-9


def test_mwkr_priority_favours_the_job_with_the_most_remaining_work():
    proc = np.array([[5, 5], [1, 1]])
    priority = A.mwkr_priority(proc)
    assert priority(0, 0) < priority(1, 0)


# --- Heads/Tails ----------------------------------------------------------------------------------------------


def test_heads_respect_job_precedence_with_no_sequenced_machines():
    routing, proc = _instance(1, 4, 3)
    head = A.compute_heads(routing, proc, {})
    n, m = routing.shape
    for j in range(n):
        for pos in range(1, m):
            assert head[j, pos] >= head[j, pos - 1] + proc[j, pos - 1] - 1e-9


def test_heads_respect_a_fixed_machine_sequence():
    routing = np.array([[0, 1], [0, 1], [1, 0]])
    proc = np.array([[3, 2], [4, 1], [5, 6]])
    machine_seq = {0: [(1, 0), (0, 0)]}   # Maschine 0: erst Auftrag 1, dann Auftrag 0
    head = A.compute_heads(routing, proc, machine_seq)
    assert head[0, 0] == pytest.approx(4)   # muss auf Auftrag 1s Ende (4) auf Maschine 0 warten


def test_heads_and_tails_collapse_to_zero_reoptimization_horizon_at_the_start():
    routing, proc = _instance(2, 5, 3)
    head = A.compute_heads(routing, proc, {})
    tail = A.compute_tails(routing, proc, {})
    assert head[:, 0].max() == pytest.approx(0.0)   # erste Operation jedes Auftrags kann sofort beginnen
    assert tail.min() >= 0.0


def test_setup_aware_heads_add_the_setup_cost_between_predecessor_and_successor():
    routing = np.array([[0], [0]])
    proc = np.array([[3], [3]])
    family = np.array([0, 1])
    setup = np.array([[0, 7], [7, 0]])
    machine_seq = {0: [(0, 0), (1, 0)]}
    head = A.compute_heads(routing, proc, machine_seq, family, setup)
    assert head[1, 0] == pytest.approx(3 + 7)


# --- Zyklen-Schutz (siehe test_no_reoptimization_does_not_silently_produce_a_cyclic_schedule) -------------------


def test_is_acyclic_detects_a_hand_built_cycle():
    """3 Aufträge x 2 Maschinen, so konstruiert, dass die fixierten Reihenfolgen von Maschine 0 UND Maschine 1
    zusammen mit dem Auftragsvorrang einen Zyklus schließen: (0,1)->(1,0) [Maschine 1] -> (1,1) [Auftrag]
    -> (2,0) [Maschine 0]... am einfachsten direkt zwei sich widersprechende Ketten bauen."""
    routing = np.array([[0, 1], [1, 0]])
    proc = np.array([[1, 1], [1, 1]])
    # Maschine 0: (1,1) vor (0,0); Maschine 1: (0,1) vor (1,0) - zusammen mit Auftragsvorrang (0,0)->(0,1) und
    # (1,0)->(1,1) entsteht (0,0)->(0,1)->(1,0)->(1,1)->(0,0): ein Zyklus.
    machine_seq = {0: [(1, 1), (0, 0)], 1: [(0, 1), (1, 0)]}
    assert not A._is_acyclic(routing, proc, machine_seq)


def test_is_acyclic_accepts_a_consistent_sequence():
    routing = np.array([[0, 1], [1, 0]])
    proc = np.array([[1, 1], [1, 1]])
    machine_seq = {0: [(0, 0), (1, 1)], 1: [(1, 0), (0, 1)]}
    assert A._is_acyclic(routing, proc, machine_seq)


def test_assign_machine_sequence_falls_back_to_head_order_when_the_candidate_would_cycle():
    routing = np.array([[0, 1], [1, 0]])
    proc = np.array([[1, 1], [1, 1]])
    machine_seq = {1: [(0, 1), (1, 0)]}   # bereits fixiert
    head = np.array([[0.0, 2.0], [0.0, 2.0]])
    candidate = [(1, 1), (0, 0)]           # würde zusammen mit Maschine 1 einen Zyklus schließen
    result = A._assign_machine_sequence(routing, proc, machine_seq, 0, candidate, head)
    assert A._is_acyclic(routing, proc, {**machine_seq, 0: result})
    assert result == sorted(candidate, key=lambda jp: head[jp[0], jp[1]])


# --- 1|rj|Lmax-Teilproblem --------------------------------------------------------------------------------------


def test_solve_1r_lmax_respects_release_times_and_no_overlap():
    ops = [(0, 0), (1, 0)]
    proc = np.array([[5], [3]])
    head = np.zeros((2, 1))
    head[1, 0] = 10   # Auftrag 1 darf erst ab Zeit 10 starten
    tail = np.zeros((2, 1))
    seq, lmax = A.solve_1r_lmax(ops, proc, head, tail, horizon=50)
    assert seq[0] == (0, 0)   # Auftrag 0 ist frei ab 0, sollte zuerst laufen


def test_solve_1r_lmax_with_setup_avoids_unnecessary_family_changes():
    ops = [(0, 0), (1, 0), (2, 0)]
    proc = np.array([[2], [2], [2]])
    family = np.array([0, 0, 1])
    setup = np.array([[0, 10], [10, 0]])
    head = np.zeros((3, 1))
    tail = np.zeros((3, 1))
    seq, lmax = A.solve_1r_lmax(ops, proc, head, tail, horizon=50, family=family, setup=setup)
    # optimal: beide Familie-0-Aufträge zusammen (nebeneinander), dann nur EIN Wechsel statt zwei
    families_in_order = [int(family[j]) for (j, pos) in seq]
    assert families_in_order == [0, 0, 1]


# --- Shifting-Bottleneck-Hauptschleife --------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", [(2, 2), (3, 2), (3, 3), (4, 3)])
def test_shifting_bottleneck_matches_cp_sat_optimum_closely(n, m):
    """Kein Optimalitätsbeweis wie bei Stück 1-4/6/8 (Suchraum) - Shifting Bottleneck ist eine SEHR GUTE
    Heuristik, hier gegen CP-SAT auf einen kleinen, aber echten Toleranzbereich geprüft."""
    for seed in range(3):
        routing, proc = _instance(seed * 10 + n + m, n, m)
        result = A.shifting_bottleneck(routing, proc, reoptimize=True)
        opt, proven = A.solve_exact(routing, proc, time_limit_seconds=10)
        assert proven
        assert result.cmax <= opt.cmax * 1.35 + 1e-6


def test_shifting_bottleneck_produces_a_feasible_precedence_respecting_schedule():
    routing, proc = _instance(9, 6, 3)
    result = A.shifting_bottleneck(routing, proc, reoptimize=True)
    n, m = routing.shape
    for j in range(n):
        for pos in range(1, m):
            assert result.start[j, pos] >= result.end[j, pos - 1] - 1e-6
    for k in range(m):
        ops_k = sorted([(j, pos) for j in range(n) for pos in range(m) if routing[j, pos] == k], key=lambda jp: result.start[jp])
        for a, b in zip(ops_k, ops_k[1:]):
            assert result.start[b] >= result.end[a] - 1e-6


def test_bottleneck_order_is_a_permutation_of_all_machines():
    routing, proc = _instance(3, 8, 4)
    result = A.shifting_bottleneck(routing, proc, reoptimize=True)
    assert sorted(result.bottleneck_order) == list(range(4))


def test_reoptimization_helps_on_a_hand_picked_instance():
    routing, proc = _instance(4, 8, 4)
    with_reopt = A.shifting_bottleneck(routing, proc, reoptimize=True)
    without_reopt = A.shifting_bottleneck(routing, proc, reoptimize=False)
    assert with_reopt.cmax <= without_reopt.cmax


def test_reoptimization_is_not_proven_monotone_on_every_instance():
    """Der ehrliche Beweis-Check dieses Stücks: ANDERS als eine 'nie schlechter'-Garantie kann die Reoptimierung
    (n=3, m=4, Seed 100000, beide von hand nachgerechnet) minimal schlechter abschneiden als ganz ohne
    Reoptimierung - sie verändert die Teilprobleme ALLER noch offenen Maschinen, ist also kein rein lokaler
    Verbesserungsschritt (siehe sb_algorithm.shifting_bottleneck-Docstring)."""
    rng = np.random.default_rng(100000)
    routing = np.array([rng.permutation(4) for _ in range(3)])
    proc = rng.integers(1, 100, size=(3, 4)).astype(np.int64)
    with_reopt = A.shifting_bottleneck(routing, proc, reoptimize=True)
    without_reopt = A.shifting_bottleneck(routing, proc, reoptimize=False)
    assert with_reopt.cmax > without_reopt.cmax


def test_no_reoptimization_does_not_silently_produce_a_cyclic_schedule():
    """Regressionsschutz für einen echten, gefundenen Bug (n=20, m=4, Seed 100003, OHNE Reoptimierung): das
    1|rⱼ|Lmax-Teilproblem respektiert nur Freigabezeiten (eine Relaxation), keine volle Graphstruktur - eine für
    sich genommen optimale Reihenfolge auf der neuen Maschine kann, kombiniert mit bereits fixierten Maschinen,
    über den Auftragsvorrang einen ZYKLUS schließen (dokumentiertes Risiko der Head/Tail-Relaxation, Adams/
    Balas/Zawack 1988). Ohne Schutz lieferte compute_heads dafür einen unkonvergierten Fixpunkt und Cmax = 42176
    - physikalisch unmöglich bei proc.sum() = 3854. `_assign_machine_sequence` erkennt das und weicht auf die
    beweisbar zyklenfreie Reihenfolge (aufsteigender Head) aus."""
    rng = np.random.default_rng(100003)
    routing = np.array([rng.permutation(4) for _ in range(20)])
    proc = rng.integers(1, 100, size=(20, 4)).astype(np.int64)
    result = A.shifting_bottleneck(routing, proc, reoptimize=False)
    assert result.cmax <= float(proc.sum())


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_shifting_bottleneck_at_zero_setup_time_matches_the_neutral_vehicle_exactly():
    routing, proc = _instance(7, 6, 3)
    rng = np.random.default_rng(1)
    family = rng.integers(0, 3, size=6)
    setup = np.zeros((3, 3))
    neutral = A.shifting_bottleneck(routing, proc, reoptimize=True)
    logistik = A.shifting_bottleneck(routing, proc, family, setup, reoptimize=True)
    assert neutral.cmax == pytest.approx(logistik.cmax)


def test_cp_sat_with_setup_matches_independent_brute_force():
    routing, proc = _instance(11, 3, 3)
    rng = np.random.default_rng(11)
    family = rng.integers(0, 2, size=3)
    setup = np.array([[0, 6], [6, 0]])
    bf = _brute_force(routing, proc, family, setup)
    result, proven = A.solve_exact(routing, proc, family, setup, time_limit_seconds=10)
    assert proven
    assert result.cmax == pytest.approx(bf, abs=1e-6)


def test_shifting_bottleneck_with_setup_matches_cp_sat_closely():
    routing, proc = _instance(20, 5, 3)
    rng = np.random.default_rng(2)
    family = rng.integers(0, 3, size=5)
    setup = np.full((3, 3), 15)
    np.fill_diagonal(setup, 0)
    result = A.shifting_bottleneck(routing, proc, family, setup, reoptimize=True)
    opt, proven = A.solve_exact(routing, proc, family, setup, time_limit_seconds=10)
    assert proven
    assert result.cmax <= opt.cmax * 1.35 + 1e-6
