"""Shifting Bottleneck - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Konzepte-Linie "Klassische Scheduling-Theorie", das erste SOTA-Stück: statt EINER
Prioritätsregel für alle Maschinen zugleich (Giffler-Thompson mit MWKR, Stück 8) sequenziert das
Shifting-Bottleneck-Verfahren (Adams, Balas & Zawack 1988) die Maschinen NACHEINANDER - jeweils die
engpassträchtigste zuerst (gelöst über ein 1|rⱼ|Lmax-Teilproblem je Maschine) - und REOPTIMIERT bereits
sequenzierte Maschinen, sobald eine neue dazukommt. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sb_constants as C
from sb_evaluation import Settings, SWEEP_LABELS, analyse, reoptimization_check, run_config, setup_gap_sweep, sweep, timing_sweep
from sb_presets import KEPT, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, seed_widget, sync_query_params
from sb_visualization import build_jobs_chart, build_machine_finish_comparison, build_machine_gantt, build_reopt_chart, build_setup_gap, build_sweep, build_timing

st.set_page_config(page_title="Shifting Bottleneck – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _reopt_check():
    return reoptimization_check()


@st.cache_data(show_spinner=False)
def _timing(m):
    return timing_sweep(m=m)


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, m, n_families):
    return setup_gap_sweep(n=n, m=m, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _fmt_pct(x):
    """Vorzeichen-korrekt: `+2.4 %` (Vergleich schlechter als Shifting Bottleneck) oder `-x %` (der Vergleich war
    hier besser - kein bewiesener Widerspruch, siehe 🚧-Abschnitt)."""
    return f"{x:+.1f} %"


st.title("🔧 Shifting Bottleneck – Engpässe zuerst, dann nachbessern")
st.markdown(
    r"""
**Erstes SOTA-Stück dieser Linie.** Giffler-Thompson mit MWKR (Stück 8) wendet EINE Prioritätsregel auf ALLE
Maschinen zugleich an. Das **Shifting-Bottleneck-Verfahren** (Adams, Balas & Zawack 1988) geht anders vor: es
sequenziert die Maschinen NACHEINANDER - bei jedem Schritt wird für jede noch offene Maschine ein eigenes,
kleines Teilproblem gelöst (`1|rⱼ|Lmax`: eine Maschine, Freigabezeiten, maximale Verspätung minimieren), die
Maschine mit dem GRÖSSTEN Lmax ist der eigentliche Engpass und wird fest sequenziert. Danach werden alle bereits
sequenzierten Maschinen **reoptimiert** - eine Idee, die dem Verfahren seinen Namen gibt.
"""
)
st.caption(
    "Neuntes Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - dasselbe Job-Shop-Modell wie Stück 8. "
    "Zwei Vehikel: **Neutral** und **Werkstatt/Logistik** (Familien mit Rüstzeit beim Wechsel je Maschine) - "
    "der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert Shifting Bottleneck", expanded=True):
    st.markdown(
        r"""
1. **Heads/Tails berechnen.** Für jede Operation die früheste Startzeit (Head, über Auftragsvorrang PLUS bereits
   festgelegte Maschinenreihenfolgen) und die späteste "Restlaufzeit" bis zu einer virtuellen Senke (Tail) -
   NUR mit den bisher sequenzierten Maschinen.
2. **Teilproblem je offener Maschine lösen.** `1|rⱼ|Lmax`: Freigabezeit = Head, Fälligkeit = Horizont − Tail,
   Ziel: maximale Verspätung minimieren - gelöst über CP-SAT (bewusste Vereinfachung statt Carliers 1982
   spezialisiertem Algorithmus, mit dem Nutzer abgestimmt).
3. **Engpass fest sequenzieren.** Die Maschine mit dem GRÖSSTEN gelösten Lmax ist der Engpass - ihre Reihenfolge
   wird permanent übernommen.
4. **Reoptimieren.** Jede bereits sequenzierte Maschine wird mit den aktualisierten Heads/Tails erneut gelöst -
   das hilft im Mittel deutlich, ist aber (ehrlicher Befund, siehe 🚧 unten) NICHT bewiesen monoton.
5. **Wiederholen**, bis alle Maschinen sequenziert sind.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.EXACT_MAX_N} löst CP-SAT das volle Problem exakt mit.")
    m_machines = st.slider("Maschinen", *bounds("m_slider"), key="m_slider",
                            help="Jeder Auftrag besucht jede Maschine genau einmal, aber in eigener Reihenfolge.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: nur Bearbeitungszeiten. Werkstatt/Logistik: dieselben Aufträge, zusätzlich in Familien mit Rüstzeit beim Wechsel je Maschine.")
    if vehicle == "logistik":
        seed_widget("setup_time_slider")
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert strukturell exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        st.session_state[KEPT["setup_time_slider"]] = setup_time
        seed_widget("n_families_slider")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider",
                                help="Weniger Familien bei gleicher Auftragszahl bedeutet mehr Wechsel und damit mehr Rüstzeit insgesamt.")
        st.session_state[KEPT["n_families_slider"]] = n_families
    else:
        setup_time = int(st.session_state.get(KEPT["setup_time_slider"], C.DEFAULT_SETUP_TIME))
        n_families = int(st.session_state.get(KEPT["n_families_slider"], C.DEFAULT_N_FAMILIES))
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Routing und Bearbeitungszeiten.")

sync_query_params({"n_slider": int(n_jobs), "m_slider": int(m_machines), "seed_input": int(seed),
                    "vehicle_radio": vehicle, "setup_time_slider": int(setup_time), "n_families_slider": int(n_families)})

settings = Settings(int(n_jobs), int(m_machines), int(seed), vehicle=vehicle, setup_time=int(setup_time), n_families=int(n_families))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
routing, proc = inst.routing, inst.proc
data_key = settings

# --- Shifting Bottleneck in Aktion -----------------------------------------------------------------------------

st.markdown("## 🎯 Shifting Bottleneck in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Engpässe nacheinander", 3: "3 · Ergebnis"}
if "sb_step" not in st.session_state or st.session_state.get("sb_step_owner") != data_key:
    st.session_state["sb_step"] = 1
    st.session_state["sb_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="sb_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    seq_col, _ = st.columns([5, 2])
    with seq_col:
        upto = st.slider("Bereits als Engpass sequenzierte Maschinen", 1, int(m_machines), value=int(m_machines), key="sb_upto")
else:
    upto = int(m_machines)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (gestapelt in eigener Maschinenreihenfolge, Farbe nach Maschine)")
        st.plotly_chart(build_jobs_chart(routing, proc), width="stretch", key="s1_jobs")
    elif step == 2:
        sequenced = set(a.sb.bottleneck_order[:upto])
        st.markdown(f"**{upto} von {int(m_machines)} Maschinen als Engpass sequenziert** (Farbe nach Auftrag)")
        st.plotly_chart(build_machine_gantt(routing, proc, a.sb.start, a.sb.end, int(m_machines), sequenced_machines=sequenced), width="stretch", key=f"s2_sched_{upto}")
    else:
        st.markdown("**Fertigstellung je Maschine: Shifting Bottleneck gegen MWKR**")
        st.plotly_chart(build_machine_finish_comparison(a.sb.machine_finish, a.mwkr.machine_finish), width="stretch", key="s3_finish")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(proc.min())} und {int(proc.max())} Minuten (Seed {seed}). Jeder Auftrag besucht jede Maschine genau einmal, aber in eigener Reihenfolge.")
elif step == 2:
    order_str = " → ".join(f"M{k + 1}" for k in a.sb.bottleneck_order[:upto])
    st.caption(f"Engpass-Reihenfolge (so wie das Verfahren sie tatsächlich gewählt hat): {order_str}. Jede Zeile ist eine Maschine, jeder Balken eine Operation, Farbe = Auftrag.")
else:
    st.caption(f"Shifting Bottleneck: Cmax {_fmt_int(a.sb.cmax)}. MWKR: {_fmt_int(a.mwkr.cmax)} (Differenz {_fmt_pct(a.gap_mwkr)}). Die gestrichelten Linien markieren jeweils die höchste Last (= Cmax).")

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was Reoptimierung und das Verfahren selbst bringen")
vehicle_note = " Auf dem Werkstatt/Logistik-Vehikel zählt die Rüstzeit je Maschine mit - alle Zahlen hier berücksichtigen sie." if vehicle == "logistik" else ""
st.caption(f"**Abstand:** Cmax eines Vergleichs gegenüber Shifting Bottleneck in Prozent - kann negativ werden, weder die Reoptimierung noch das Verfahren insgesamt sind bewiesen optimal.{vehicle_note}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Shifting Bottleneck (Cmax)", _fmt_int(a.sb.cmax), help="Die Zielgröße: Gesamtdurchlaufzeit mit Shifting Bottleneck (mit Reoptimierung), auf dem gewählten Vehikel.")
m2.metric("Ohne Reoptimierung", _fmt_pct(a.gap_no_reopt), delta_color="off", help="Derselbe Ablauf, aber ohne Schritt 4 (Reoptimierung bereits sequenzierter Maschinen).")
m3.metric("MWKR (Stück 8)", _fmt_pct(a.gap_mwkr), delta_color="off", help="Die beste einfache Prioritätsregel aus dem Vorgängerstück - hier die Messlatte für ein ganzes VERFAHREN statt einer einzelnen Regel.")
if a.optimal is not None and a.optimal_proven:
    m4.metric("CP-SAT (exakte Gegenprobe)", "trifft das Optimum exakt" if a.sb_matches_optimum else f"{a.sb_ratio_to_optimum:.3f}× Optimum", delta_color="off",
              help="OR-Tools CP-SAT hat diese Instanz bewiesen exakt gelöst (volles Modell, nicht nur ein Teilproblem).")
elif a.optimal is not None:
    m4.metric("CP-SAT", "Zeitlimit erreicht", delta_color="off", help="CP-SAT hat innerhalb des Zeitlimits keine bewiesen optimale Lösung gefunden.")
else:
    m4.metric("CP-SAT", f"erst ab n ≤ {C.EXACT_MAX_N}", delta_color="off", help="Bei dieser Größe wäre eine exakte Lösung des vollen Modells aussichtslos.")

if a.gap_no_reopt < 0 or a.gap_mwkr < 0:
    candidates = [("die Fassung ohne Reoptimierung", a.gap_no_reopt), ("MWKR", a.gap_mwkr)]
    worse_than, worst_gap = min(candidates, key=lambda c: c[1])
    vehicle_hint = " (auch mit Rüstzeiten möglich, die Teilprobleme kennen sie zwar, das Verfahren bleibt aber eine Heuristik)" if vehicle == "logistik" else ""
    st.warning(f"⚠️ Shifting Bottleneck (mit Reoptimierung) schneidet hier sogar schlechter ab als {worse_than}: {abs(worst_gap):.1f} % mehr{vehicle_hint}. Kein Fehler - die Reoptimierung verändert die Teilprobleme ALLER noch offenen Maschinen und ist deshalb kein rein lokaler, monoton verbessernder Schritt (siehe 🔬 unten). Das kann auch OHNE Rüstzeiten vorkommen.")
else:
    tail = " (auch mit Rüstzeiten - bei dieser Instanz schlägt Shifting Bottleneck MWKR trotzdem deutlich)" if vehicle == "logistik" else ""
    st.success(f"✅ Shifting Bottleneck ist {a.gap_no_reopt:.1f} % besser als ohne Reoptimierung und {a.gap_mwkr:.1f} % besser als MWKR{tail}.")

st.markdown("---")

# --- Sweeps --------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004). Die gepunktete Nulllinie markiert Gleichstand mit Shifting Bottleneck.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------

st.subheader("🔬 Hilft die Reoptimierung wirklich - und wie oft?")
if st.button("Reoptimierung über n = 3 bis 8 auswerten (dauert etwa 10 Sekunden)", key="reopt_start"):
    st.session_state["reopt_on"] = True
if st.session_state.get("reopt_on"):
    rows_r = _reopt_check()
    st.plotly_chart(build_reopt_chart(rows_r), width="stretch", key="reopt_chart")
    st.caption("Grün: Anteil der Instanzen, bei denen die Reoptimierung mindestens gleich gut wie ohne war - KEINE 100 %-Garantie (anders als der Suchraum-Satz in Stück 8). Blau gestrichelt: wie oft Shifting Bottleneck allein das CP-SAT-Optimum trifft.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine exakte Lösung wirklich?")
if st.button("Rechenzeit für n = 2 bis 8 messen (dauert etwa 1 Sekunde)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing(int(m_machines))
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht das volle CP-SAT-Modell bereits {last['exact_seconds']*1000:.0f} ms, Shifting Bottleneck (viele kleine Teilprobleme) {last['sb_seconds']*1000:.1f} ms.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt der Vorsprung, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert wenige Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.EXACT_MAX_N), int(m_machines), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("Shifting Bottleneck (mit Reoptimierung, rüstzeitbewusst über das Teilproblem) verglichen mit der echten Optimallösung MIT Rüstzeiten (CP-SAT, deshalb kleine Instanz).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Reoptimierung ist eine bewiesene Verbesserung** | Sie verändert die Teilprobleme ALLER noch offenen Maschinen und ist deshalb kein rein lokaler Schritt - in seltenen Fällen (siehe 🔬 oben) minimal schlechter als ganz ohne Reoptimierung. | Kein direkter Nachfolger in dieser Linie |
| **Das 1\\|rⱼ\\|Lmax-Teilproblem wird exakt gelöst** | Hier über CP-SAT statt Carliers (1982) spezialisiertem Branch-and-Bound - eine bewusste Vereinfachung, für die Größenordnungen dieser Demo ohne praktischen Unterschied. | Kein direkter Nachfolger in dieser Linie |
| **Eine feste Maschinenreihenfolge reicht** | Für sehr harte Instanzen brauchen professionelle Löser lokale Suche über GANZE Zeitpläne, nicht nur eine schrittweise Konstruktion. | **Job-Shop-Tabu-Search** (nächstes SOTA-Stück) |
"""
)
st.caption(
    "Neuntes Stück der Linie „Klassische Scheduling-Theorie“, erstes SOTA-Stück. Verwandt: die Trajektorien-"
    "Metaheuristiken-Linie (Tabu Search, lokale Suche über ganze Zeitpläne)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** (Job Shop, $C_{\max}$): dasselbe Modell wie Stück 8 - $n$ Aufträge, $m$ Maschinen, jeder Auftrag mit
eigener Reihenfolge über die Maschinen.

**Heads/Tails.** $\text{head}(j,\text{pos})$ = längster Weg vom Start zu Operation $(j,\text{pos})$ über
Auftragsvorrang PLUS bereits fest sequenzierte Maschinen (inklusive Rüstzeit bei Familienwechsel, falls Vehikel
B). $\text{tail}(j,\text{pos})$ symmetrisch bis zu einer virtuellen Senke.

**Teilproblem** ($1|r_j|L_{\max}$ bzw. $1|r_j,s_{jk}|L_{\max}$): für eine Maschine $k$, Freigabezeit
$r_j = \text{head}(j,\text{pos})$, Fälligkeit $d_j = H - \text{tail}(j,\text{pos})$ (Horizont $H$), gesucht die
Reihenfolge, die $\max_j (C_j - d_j)$ minimiert.

**Engpasswahl.** Unter allen noch offenen Maschinen die mit dem GRÖSSTEN gelösten $L_{\max}$ - ihre Reihenfolge
wird fest übernommen.

**Reoptimierung.** Jede bereits sequenzierte Maschine $k$ wird erneut als $1|r_j|L_{\max}$-Teilproblem gelöst,
diesmal OHNE sich selbst im Graphen und mit aktualisierten Heads/Tails der neu hinzugekommenen Maschine.

**CP-SAT-Modell** (`solve_exact`): dasselbe Kreis-Modell wie Stück 8, hier als exakte Gegenprobe auf das VOLLE
Problem (nicht nur ein Teilproblem).

Implementiert in `sb_algorithm.py` (Heads/Tails, Teilproblem-Löser, Shifting-Bottleneck-Hauptschleife,
Giffler-Thompson/MWKR als Vergleichsbasis, CP-SAT), `sb_scenario.py`/`sb_scenario_logistik.py` (die zwei
Vehikel), `sb_evaluation.py` (Kennzahlen, Sweep, Reoptimierungs-Check, Timing-Messreihe, Rüstzeit-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
