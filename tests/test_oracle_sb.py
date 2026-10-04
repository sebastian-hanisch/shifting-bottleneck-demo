"""Unabhängiges Orakel für Shifting Bottleneck: Heads/Tails per networkx-DAG (längste Wege), das Teilproblem
1|rⱼ|Lmax per Vollaufzählung aller Reihenfolgen, und das Gesamtergebnis gegen eine Vollaufzählung aller
Maschinenfolgen (Cmax-Untergrenze) sowie eine Neuberechnung von Cmax aus den abgeleiteten Maschinenfolgen."""

import itertools
import random

import numpy as np
import pytest

import sb_algorithm as A
import sb_scenario as S
import sb_scenario_logistik as SL

nx = pytest.importorskip("networkx")


def _graph(n, m, proc, machine_seq, fam, setup):
    G = nx.DiGraph()
    for j in range(n):
        for p in range(m):
            G.add_node((j, p))
            if p:
                G.add_edge((j, p - 1), (j, p), w=int(proc[j, p - 1]))
    for seq in machine_seq.values():
        for a, b in zip(seq, seq[1:]):
            G.add_edge(a, b, w=int(proc[a]) + (int(setup[fam[a[0]]][fam[b[0]]]) if fam is not None else 0))
    return G


def _longest_paths(n, m, proc, machine_seq, fam, setup):
    G = _graph(n, m, proc, machine_seq, fam, setup)
    order = list(nx.topological_sort(G))
    head = {v: 0 for v in G}
    for v in order:
        head[v] = max([head[u] + G[u][v]["w"] for u in G.predecessors(v)], default=0)
    tail = {v: 0 for v in G}
    for v in reversed(order):
        tail[v] = max([G[v][u]["w"] - int(proc[v]) + int(proc[u]) + tail[u] for u in G.successors(v)], default=0)
    return head, tail, G


def _instance(rng, setups):
    n, m, seed = rng.randint(2, 4), rng.randint(2, 3), rng.randint(0, 10**6)
    m = 2 if n == 4 else m                                   # Vollaufzählung (n!)^m klein halten
    if setups:
        li = SL.generate(n, m, seed, n_families=rng.randint(2, 3), setup_time=rng.choice([0, 10, 40]))
        return n, m, li.routing, li.proc, li.family, li.setup
    inst = S.generate(n, m, seed)
    return n, m, inst.routing, inst.proc, None, None


def _brute_force_cmax(n, m, routing, proc, fam, setup):
    best = None
    for combo in itertools.product(itertools.permutations(range(n)), repeat=m):
        seq = {k: [(j, int(np.where(routing[j] == k)[0][0])) for j in combo[k]] for k in range(m)}
        try:
            _, _, G = _longest_paths(n, m, proc, seq, fam, setup)
        except nx.NetworkXUnfeasible:
            continue
        head = {}
        for v in nx.topological_sort(G):
            head[v] = max([head[u] + G[u][v]["w"] for u in G.predecessors(v)], default=0)
        c = max(head[v] + int(proc[v]) for v in G)
        best = c if best is None else min(best, c)
    return best


def test_heads_tails_and_subproblem_match_independent_computation():
    rng = random.Random(11)
    for it in range(40):
        n, m, routing, proc, fam, setup = _instance(rng, it % 2)
        g = A.giffler_thompson(routing, proc, lambda j, p, r=np.random.default_rng(it).permutation(n): r[j], fam, setup)
        seqs = {k: [(j, p) for _, j, p in sorted((g.start[j, p], j, p) for j in range(n) for p in range(m) if routing[j, p] == k)]
                for k in range(m)}
        fixed = {k: seqs[k] for k in rng.sample(range(m), rng.randint(0, m - 1))}
        head, tail = A.compute_heads(routing, proc, fixed, fam, setup), A.compute_tails(routing, proc, fixed, fam, setup)
        hh, tt, _ = _longest_paths(n, m, proc, fixed, fam, setup)
        assert all(head[v] == hh[v] and tail[v] == tt[v] for v in hh)
        k = rng.choice([x for x in range(m) if x not in fixed])
        ops = [(j, p) for j in range(n) for p in range(m) if routing[j, p] == k]
        horizon = int(proc.sum()) + (int(setup.max()) * n * m if setup is not None else 0) + 1

        def lmax(order):
            t, prev, mx = 0, None, -10**9
            for (j, p) in order:
                s = int(head[j, p])
                if prev is not None:
                    s = max(s, t + (int(setup[fam[prev[0]]][fam[j]]) if fam is not None else 0))
                t = s + int(proc[j, p])
                mx, prev = max(mx, t - (horizon - int(tail[j, p]))), (j, p)
            return mx

        seq, value = A.solve_1r_lmax(ops, proc, head, tail, horizon, fam, setup)
        assert value == min(lmax(pm) for pm in itertools.permutations(ops)) == lmax(seq)


def test_shifting_bottleneck_is_feasible_and_never_beats_the_brute_force_optimum():
    rng = random.Random(12)
    for it in range(20):
        n, m, routing, proc, fam, setup = _instance(rng, it % 2)
        best = _brute_force_cmax(n, m, routing, proc, fam, setup)
        for reopt in (True, False):
            r = A.shifting_bottleneck(routing, proc, fam, setup, reoptimize=reopt)
            seqs = {k: [(j, p) for _, j, p in sorted((r.start[j, p], j, p) for j in range(n) for p in range(m) if routing[j, p] == k)]
                    for k in range(m)}
            head, _, G = _longest_paths(n, m, proc, seqs, fam, setup)          # wirft bei Zyklus
            assert r.cmax == max(head[v] + int(proc[v]) for v in G) and r.cmax >= best
            assert all(r.start[v] == head[v] for v in G) and sorted(r.bottleneck_order) == list(range(m))
