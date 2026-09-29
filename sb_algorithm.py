"""Shifting-Bottleneck-Verfahren (Adams, Balas & Zawack 1988, Management Science 34(3):391-401) - das erste
SOTA-Stück dieser Linie: statt EINER Prioritätsregel für alle Maschinen zugleich (Giffler-Thompson, Stück 8)
sequenziert Shifting Bottleneck die Maschinen NACHEINANDER, jeweils die "engpassträchtigste" zuerst, und
REOPTIMIERT bereits sequenzierte Maschinen, sobald eine neue dazukommt.

Vier Bausteine:
1. **Heads/Tails** (`compute_heads`/`compute_tails`): längster Weg vom Start zu jeder Operation (Head, über
   Auftragsvorrang PLUS bereits festgelegte Maschinenreihenfolgen) bzw. von jeder Operation zu einer virtuellen
   Senke (Tail) - liefert Freigabezeit und "verbleibende Zeit bis zum Ende" je Operation, nur mit den bisher
   sequenzierten Maschinen.
2. **Teilproblem je unsequenzierter Maschine** (`solve_1r_lmax`): `1|rⱼ|Lmax` (Neutral) bzw. `1|rⱼ,sⱼₖ|Lmax`
   (Werkstatt/Logistik, sequenzabhängige Rüstzeit) - Freigabezeiten = Heads, Fälligkeiten = Horizont − Tails,
   Ziel: maximale Verspätung minimieren. Bewusste Vereinfachung (mit dem Nutzer abgestimmt): CP-SAT statt
   Carliers (1982) spezialisiertem Branch-and-Bound - dieselbe "CP-SAT als exaktes Standardwerkzeug"-Linie wie
   in jedem anderen Stück dieses Portfolios.
3. **Engpasswahl**: die Maschine mit dem GRÖSSTEN gelösten Lmax ist der Engpass - ihre Reihenfolge wird fest
   übernommen.
4. **Reoptimierung**: jede BEREITS sequenzierte Maschine wird erneut gelöst (ohne sich selbst, mit den
   aktualisierten Heads/Tails der neu hinzugekommenen Maschine) - das ist der Unterschied zu naivem "eine
   Maschine nach der anderen fest einplanen" und hilft im Mittel deutlich (siehe README/Messreihe). Ehrlicher
   Befund: da die Reoptimierung die Heads/Tails ALLER noch offenen Maschinen verändert, ist sie KEIN rein
   lokaler, monoton verbessernder Schritt - in seltenen Fällen schneidet sie minimal schlechter ab als ganz ohne
   Reoptimierung (Experiment unten).

Giffler-Thompson/MWKR (Stück 8, hier als Vergleichsbasis kopiert, nicht importiert - jedes Stück dieser Linie ist
eigenständig lauffähig) und das volle CP-SAT-Kreis-Modell (exakte Gegenprobe für kleine n) sind unverändert."""

import math
import os
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

NUM_SEARCH_WORKERS = min(8, os.cpu_count() or 1)  # NIE hart auf eine Zahl setzen - siehe project memory
# (weighted-tardiness-demo brach auf einem 4-Kern-CI-Runner durch Oversubscription bei hart kodierten 8 Workern)


@dataclass
class Result:
    start: np.ndarray          # (n, m): Startzeit je Operation
    end: np.ndarray            # (n, m): Fertigstellung je Operation
    cmax: float
    machine_finish: np.ndarray  # (m,): Fertigstellung der letzten Operation je Maschine
    bottleneck_order: object = None  # nur bei shifting_bottleneck: Maschinen in der Reihenfolge, in der sie zum Engpass wurden


# --- Giffler-Thompson/MWKR (Stück 8 dieser Linie, Vergleichsbasis) ----------------------------------------------


def _schedulable(routing, job_ptr):
    n, m = routing.shape
    return [(j, job_ptr[j]) for j in range(n) if job_ptr[j] < m]


def _setup_cost(family, setup, prev_family_on_machine, j):
    if family is None or prev_family_on_machine is None:
        return 0
    return int(setup[prev_family_on_machine, family[j]])


def giffler_thompson(routing, proc, priority, family=None, setup=None):
    """Konstruiert EINEN aktiven Zeitplan per Prioritätsregel bei jedem Konflikt - unverändert aus Stück 8, hier
    als "was bringt ein besseres VERFAHREN gegenüber der besten einfachen REGEL" Vergleichsbasis."""
    n, m = routing.shape
    job_ptr = [0] * n
    job_free = np.zeros(n)
    machine_free = np.zeros(m)
    prev_family_on_machine = [None] * m
    start = np.zeros((n, m))
    end = np.zeros((n, m))
    for _ in range(n * m):
        ops = _schedulable(routing, job_ptr)
        best = None
        for (j, pos) in ops:
            k = routing[j, pos]
            s = _setup_cost(family, setup, prev_family_on_machine[k], j)
            st = max(job_free[j], machine_free[k] + s)
            ct = st + proc[j, pos]
            if best is None or ct < best[0]:
                best = (ct, k)
        cstar, mstar = best
        conflict = []
        for (j, pos) in ops:
            k = routing[j, pos]
            if k != mstar:
                continue
            s = _setup_cost(family, setup, prev_family_on_machine[k], j)
            st = max(job_free[j], machine_free[k] + s)
            if st < cstar:
                conflict.append((j, pos))
        j, pos = min(conflict, key=lambda jp: priority(jp[0], jp[1]))
        k = routing[j, pos]
        s = _setup_cost(family, setup, prev_family_on_machine[k], j)
        st = max(job_free[j], machine_free[k] + s)
        start[j, pos], end[j, pos] = st, st + proc[j, pos]
        job_free[j] = end[j, pos]
        machine_free[k] = end[j, pos]
        if family is not None:
            prev_family_on_machine[k] = family[j]
        job_ptr[j] += 1
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end[j, pos])
    return Result(start, end, float(end.max()), machine_finish)


def mwkr_priority(proc):
    """Most Work Remaining - die beste einfache Regel aus Stück 8. Bleibt hier als Vergleichsbasis: 'was bringt
    ein VERFAHREN (Shifting Bottleneck) gegenüber der besten einzelnen REGEL'."""
    def key(j, pos):
        return -float(proc[j, pos:].sum())
    return key


# --- Heads/Tails: längster Weg über Auftragsvorrang UND bereits sequenzierte Maschinen --------------------------


def compute_heads(routing, proc, machine_seq, family=None, setup=None):
    """Head[j,pos] = früheste Startzeit dieser Operation, gegeben Auftragsvorrang (vorherige Operation desselben
    Auftrags) UND die Reihenfolge auf JEDER BEREITS SEQUENZIERTEN Maschine (inklusive Rüstzeit beim
    Familienwechsel, falls Vehikel B). `machine_seq`: {Maschine: Liste von (j,pos) in Sequenzreihenfolge}, NUR für
    bereits sequenzierte Maschinen - eine Fixpunkt-Iteration über den (azyklischen) Graphen aus beiden
    Vorrangarten reicht, da er höchstens n·m Knoten hat."""
    n, m = routing.shape
    head = np.zeros((n, m))
    pred_on_machine = {}
    pred_setup = {}
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            pred_on_machine[seq[idx]] = seq[idx - 1]
            if family is not None:
                pj, pp = seq[idx - 1]
                j2, p2 = seq[idx]
                pred_setup[seq[idx]] = int(setup[family[pj], family[j2]])
    for _ in range(n * m + 2):
        changed = False
        for j in range(n):
            for pos in range(m):
                val = 0.0
                if pos > 0:
                    val = max(val, head[j, pos - 1] + proc[j, pos - 1])
                pred = pred_on_machine.get((j, pos))
                if pred is not None:
                    pj, pp = pred
                    s = pred_setup.get((j, pos), 0)
                    val = max(val, head[pj, pp] + proc[pj, pp] + s)
                if val != head[j, pos]:
                    head[j, pos] = val
                    changed = True
        if not changed:
            break
    return head


def compute_tails(routing, proc, machine_seq, family=None, setup=None):
    """Spiegelbild von `compute_heads`: Tail[j,pos] = längster Weg von DIESER Operation bis zu einer virtuellen
    Senke (über die folgenden Operationen desselben Auftrags UND die Nachfolger auf bereits sequenzierten
    Maschinen) - zusammen mit einem Horizont ergibt das eine Fälligkeit je Operation für das 1|rⱼ|Lmax-Teilproblem."""
    n, m = routing.shape
    tail = np.zeros((n, m))
    succ_on_machine = {}
    succ_setup = {}
    for seq in machine_seq.values():
        for idx in range(len(seq) - 1):
            succ_on_machine[seq[idx]] = seq[idx + 1]
            if family is not None:
                j1, p1 = seq[idx]
                sj, sp = seq[idx + 1]
                succ_setup[seq[idx]] = int(setup[family[j1], family[sj]])
    for _ in range(n * m + 2):
        changed = False
        for j in range(n):
            for pos in range(m - 1, -1, -1):
                val = 0.0
                if pos < m - 1:
                    val = max(val, tail[j, pos + 1] + proc[j, pos + 1])
                succ = succ_on_machine.get((j, pos))
                if succ is not None:
                    sj, sp = succ
                    s = succ_setup.get((j, pos), 0)
                    val = max(val, tail[sj, sp] + proc[sj, sp] + s)
                if val != tail[j, pos]:
                    tail[j, pos] = val
                    changed = True
        if not changed:
            break
    return tail


# --- Teilproblem je Maschine: 1|rⱼ|Lmax (Neutral) bzw. 1|rⱼ,sⱼₖ|Lmax (Werkstatt/Logistik) -----------------------


def solve_1r_lmax(ops, proc, head, tail, horizon, family=None, setup=None, time_limit_seconds=5.0):
    """`ops`: Liste von (j,pos) auf EINER Maschine. Ohne Rüstzeit reicht ein einfaches Intervall+NoOverlap-Modell
    (Reihenfolge ist die einzige Freiheit); MIT Rüstzeit wird daraus ein Kreis-Modell (wie beim vollen
    CP-SAT-Modell je Maschine), weil die Rüstzeit von der WAHL der Reihenfolge abhängt, nicht nur von der
    Zuordnung. Gibt die gelöste Reihenfolge und das erreichte Lmax zurück."""
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    # BEWUSST 1 Worker (nicht NUM_SEARCH_WORKERS): dieses Teilproblem wird viele Male pro Lauf gelöst, und seine
    # zurückgegebene REIHENFOLGE (nicht nur der Lmax-Wert) fließt in die nächste Iteration ein - bei mehreren
    # gleichwertigen Lösungen (Bindungen) kann die parallele Suche je nach Worker-Anzahl eine ANDERE davon finden,
    # was sich durchs ganze Verfahren fortpflanzt (gefunden: derselbe Seed lieferte auf dieser Maschine mit 4 vs.
    # 8 Workern zwei verschiedene, beide gültig optimale Cmax-Werte). 1 Worker ist für diese winzigen
    # Ein-Maschinen-Teilprobleme schnell genug UND deterministisch unabhängig von der Kernzahl der Umgebung
    # (wichtig: lokal und auf Streamlit Cloud kann os.cpu_count() unterschiedlich sein).
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = time_limit_seconds

    if family is None or int(np.max(setup)) == 0:
        # Bei Rüstzeit 0 bewusst denselben (einfacheren) Modellpfad wie ganz ohne Familien nehmen - sonst
        # könnten die beiden strukturell verschiedenen CP-SAT-Modelle bei GLEICHWERTIGEN Lösungen (Lmax-Bindungen)
        # unterschiedlich tiebreaken und dadurch den späteren Verlauf des Verfahrens verändern (siehe Test:
        # Werkstatt/Logistik-Vehikel bei Rüstzeit 0 muss exakt aufs neutrale Vehikel zurückfallen).
        starts, ends, intervals = {}, {}, []
        for (j, pos) in ops:
            r, d = int(head[j, pos]), int(proc[j, pos])
            s = model.NewIntVar(r, horizon, f"s{j}_{pos}")
            e = model.NewIntVar(r, horizon, f"e{j}_{pos}")
            model.Add(e == s + d)
            starts[j, pos], ends[j, pos] = s, e
            intervals.append(model.NewIntervalVar(s, d, e, f"iv{j}_{pos}"))
        model.AddNoOverlap(intervals)
    else:
        starts, ends = {}, {}
        for (j, pos) in ops:
            r, d = int(head[j, pos]), int(proc[j, pos])
            starts[j, pos] = model.NewIntVar(r, horizon, f"s{j}_{pos}")
            ends[j, pos] = model.NewIntVar(r, horizon, f"e{j}_{pos}")
            model.Add(ends[j, pos] == starts[j, pos] + d)
        arcs = []
        for idx, (j, pos) in enumerate(ops):
            arcs.append((0, idx + 1, model.NewBoolVar(f"a0_{idx}")))
            arcs.append((idx + 1, 0, model.NewBoolVar(f"a{idx}_0")))
        for i1, (j1, p1) in enumerate(ops):
            for i2, (j2, p2) in enumerate(ops):
                if i1 == i2:
                    continue
                lit = model.NewBoolVar(f"a{i1}_{i2}")
                arcs.append((i1 + 1, i2 + 1, lit))
                s = int(setup[family[j1], family[j2]])
                model.Add(starts[j2, p2] >= ends[j1, p1] + s).OnlyEnforceIf(lit)
        model.AddCircuit(arcs)

    lmax = model.NewIntVar(-horizon, horizon, "lmax")
    for (j, pos) in ops:
        due = horizon - int(tail[j, pos])
        model.Add(lmax >= ends[j, pos] - due)
    model.Minimize(lmax)
    solver.Solve(model)
    seq = sorted(ops, key=lambda jp: solver.Value(starts[jp]))
    return seq, solver.Value(lmax)


def _is_acyclic(routing, proc, machine_seq):
    """Kahn-Algorithmus über denselben Graphen wie `compute_heads`/`compute_tails` (Auftragsvorrang + bereits
    fest sequenzierte Maschinen). Nötig, weil das 1|rⱼ|Lmax-Teilproblem NUR Freigabezeiten (untere Schranken)
    respektiert, nicht die volle Graphstruktur: eine für sich genommen optimale Reihenfolge auf der NEUEN
    Maschine kann, kombiniert mit bereits fixierten Maschinen, einen Zyklus über den Auftragsvorrang schließen -
    ein dokumentiertes Risiko der Head/Tail-Relaxation in Shifting Bottleneck (Adams/Balas/Zawack 1988; siehe
    README)."""
    n, m = routing.shape
    indeg = {(j, pos): 0 for j in range(n) for pos in range(m)}
    adj = {(j, pos): [] for j in range(n) for pos in range(m)}

    def add_edge(u, v):
        adj[u].append(v)
        indeg[v] += 1

    for j in range(n):
        for pos in range(1, m):
            add_edge((j, pos - 1), (j, pos))
    for seq in machine_seq.values():
        for idx in range(1, len(seq)):
            add_edge(seq[idx - 1], seq[idx])

    queue = [node for node, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return visited == n * m


def _assign_machine_sequence(routing, proc, machine_seq, k, candidate_seq, head):
    """Übernimmt `candidate_seq` (das Ergebnis von `solve_1r_lmax`) für Maschine `k`, AUSSER das würde einen
    Zyklus schließen (siehe `_is_acyclic`) - dann der garantiert sichere Rückfall: die Operationen dieser
    Maschine nach AUFSTEIGENDEM Head sortiert. Das ist beweisbar zyklenfrei: `head` ist der längste Weg im
    BISHERIGEN (zyklenfreien) Graphen, eine neue Kante von einem Knoten mit kleinerem zu einem mit
    größerem-oder-gleichem Head kann unmöglich einen bereits existierenden Pfad in Gegenrichtung schließen."""
    trial = dict(machine_seq)
    trial[k] = candidate_seq
    if _is_acyclic(routing, proc, trial):
        return candidate_seq
    return sorted(candidate_seq, key=lambda jp: head[jp[0], jp[1]])


# --- Shifting-Bottleneck-Hauptschleife --------------------------------------------------------------------------


def shifting_bottleneck(routing, proc, family=None, setup=None, reoptimize=True, time_limit_seconds=5.0):
    """Adams, Balas & Zawack (1988). `reoptimize=False` lässt den Reoptimierungsschritt (4) weg - reines
    Sicherheitsnetz/Vergleichsbaustein für die Messreihe. WICHTIG (ehrlicher Befund, siehe README): die
    Reoptimierung verändert die Heads/Tails der bereits sequenzierten Maschinen und damit die Teilprobleme ALLER
    noch offenen Maschinen - das ist KEIN rein lokaler Verbesserungsschritt auf einer festen Struktur, sondern
    verschiebt den weiteren Ablauf des Verfahrens selbst. Sie hilft im Mittel sehr deutlich, ist aber NICHT
    bewiesen monoton (in seltenen Fällen minimal schlechter als ganz ohne Reoptimierung - Messreihe/Experiment)."""
    n, m = routing.shape
    horizon = int(proc.sum()) + (int(setup.max()) * n * m if setup is not None else 0) + 1
    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))

    machine_seq = {}
    unsequenced = set(range(m))
    bottleneck_order = []
    while unsequenced:
        head = compute_heads(routing, proc, machine_seq, family, setup)
        tail = compute_tails(routing, proc, machine_seq, family, setup)
        best_k, best_seq, best_lmax = None, None, -math.inf
        for k in unsequenced:
            seq, lmax = solve_1r_lmax(ops_per_machine[k], proc, head, tail, horizon, family, setup, time_limit_seconds)
            if lmax > best_lmax:
                best_k, best_seq, best_lmax = k, seq, lmax
        machine_seq[best_k] = _assign_machine_sequence(routing, proc, machine_seq, best_k, best_seq, head)
        bottleneck_order.append(best_k)
        unsequenced.discard(best_k)
        if reoptimize:
            for k in list(machine_seq.keys()):
                if k == best_k:
                    continue
                trial = dict(machine_seq)
                del trial[k]
                head2 = compute_heads(routing, proc, trial, family, setup)
                tail2 = compute_tails(routing, proc, trial, family, setup)
                seq2, _ = solve_1r_lmax(ops_per_machine[k], proc, head2, tail2, horizon, family, setup, time_limit_seconds)
                machine_seq[k] = _assign_machine_sequence(routing, proc, trial, k, seq2, head2)

    head_final = compute_heads(routing, proc, machine_seq, family, setup)
    start = head_final
    end = np.zeros((n, m))
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            end[j, pos] = start[j, pos] + proc[j, pos]
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end[j, pos])
    return Result(start, end, float(end.max()), machine_finish, bottleneck_order)


# --- CP-SAT (exakte Gegenprobe, unverändert aus Stück 8) ---------------------------------------------------------


def solve_exact(routing, proc, family=None, setup=None, time_limit_seconds=15.0):
    """Ein Kreis-Modell JE MASCHINE - dieselbe exakte Gegenprobe wie in `job-shop-demo`, hier eigener Code für
    dieses Repo."""
    n, m = routing.shape
    if family is None:
        family = np.zeros(n, dtype=np.int64)
        setup = np.zeros((1, 1), dtype=np.int64)
    horizon = int(np.sum(proc)) + int(np.max(setup)) * n * m + 1

    model = cp_model.CpModel()
    start = {}
    end = {}
    for j in range(n):
        for pos in range(m):
            start[j, pos] = model.NewIntVar(0, horizon, f"s{j}_{pos}")
            end[j, pos] = model.NewIntVar(0, horizon, f"e{j}_{pos}")
            model.Add(end[j, pos] == start[j, pos] + int(proc[j, pos]))
            if pos > 0:
                model.Add(start[j, pos] >= end[j, pos - 1])

    ops_per_machine = [[] for _ in range(m)]
    for j in range(n):
        for pos in range(m):
            ops_per_machine[routing[j, pos]].append((j, pos))

    for k in range(m):
        ops = ops_per_machine[k]
        arcs = []
        for idx, (j, pos) in enumerate(ops):
            lit = model.NewBoolVar(f"a0_{k}_{idx}")
            arcs.append((0, idx + 1, lit))
            lit_back = model.NewBoolVar(f"a{k}_{idx}_0")
            arcs.append((idx + 1, 0, lit_back))
        for i2, (j1, p1_) in enumerate(ops):
            for j2idx, (j2, p2_) in enumerate(ops):
                if i2 == j2idx:
                    continue
                lit = model.NewBoolVar(f"a{k}_{i2}_{j2idx}")
                arcs.append((i2 + 1, j2idx + 1, lit))
                s = int(setup[family[j1], family[j2]])
                model.Add(start[j2, p2_] >= end[j1, p1_] + s).OnlyEnforceIf(lit)
        model.AddCircuit(arcs)

    cmax = model.NewIntVar(0, horizon, "cmax")
    model.AddMaxEquality(cmax, list(end.values()))
    model.Minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = NUM_SEARCH_WORKERS
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False

    start_arr = np.zeros((n, m))
    end_arr = np.zeros((n, m))
    machine_finish = np.zeros(m)
    for j in range(n):
        for pos in range(m):
            start_arr[j, pos] = solver.Value(start[j, pos])
            end_arr[j, pos] = solver.Value(end[j, pos])
            machine_finish[routing[j, pos]] = max(machine_finish[routing[j, pos]], end_arr[j, pos])
    result = Result(start_arr, end_arr, float(solver.Value(cmax)), machine_finish)
    return result, status == cp_model.OPTIMAL
