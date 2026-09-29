"""Konstanten der Shifting-Bottleneck-Demo: beide Vehikel (Neutral, Werkstatt/Logistik), Regler, Messreihen-Seeds.
Dasselbe Job-Shop-Modell wie `job-shop-demo` (Stück 8 dieser Linie) - Shifting Bottleneck ist ein BESSERES
Verfahren für DASSELBE Problem, kein neues Modell."""

N_MIN, N_MAX, DEFAULT_N, N_STEP = 2, 20, 10, 1
M_MIN, M_MAX, DEFAULT_M = 2, 6, 4
SEED_MAX = 999999
DEFAULT_SEED = 60
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3

# Bearbeitungszeiten
P_MIN, P_MAX = 1, 100

# CP-SAT exakte Gegenprobe (dasselbe Kreis-Modell wie job-shop-demo): praktisches Limit für eine live nutzbare
# Demo.
EXACT_MAX_N = 8
EXACT_TIME_LIMIT_SECONDS = 15.0

# Shifting Bottleneck selbst löst je Iteration bis zu m Teilprobleme (1|rj|Lmax bzw. mit Rüstzeit ein
# Kreis-Modell) plus die Reoptimierung bereits sequenzierter Maschinen - jedes Teilproblem bekommt ein eigenes,
# kurzes Zeitlimit (siehe README: in der Messreihe nie erreicht, nur als Sicherheitsnetz).
SUBPROBLEM_TIME_LIMIT_SECONDS = 5.0

# --- Vehikel B "Werkstatt/Logistik" ---------------------------------------------------------------------------
N_FAMILIES_MIN, N_FAMILIES_MAX, DEFAULT_N_FAMILIES = 2, 6, 3
SETUP_TIME_MIN, SETUP_TIME_MAX, DEFAULT_SETUP_TIME = 0, 60, 15

VEHICLE_LABELS = {"neutral": "Neutral", "logistik": "Werkstatt/Logistik"}
DEFAULT_VEHICLE = "neutral"


def _preset(n=DEFAULT_N, m=DEFAULT_M, vehicle=DEFAULT_VEHICLE, setup_time=DEFAULT_SETUP_TIME, n_families=DEFAULT_N_FAMILIES):
    return {"n": n, "m": m, "seed": DEFAULT_SEED, "vehicle": vehicle, "setup_time": setup_time, "n_families": n_families}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleine Instanz (CP-SAT sichtbar)": _preset(n=EXACT_MAX_N),
    "Große Instanz (Skalierung)": _preset(n=N_MAX),
    "Werkstatt/Logistik-Vehikel": _preset(vehicle="logistik"),
    "Hohe Rüstlast (Werkstatt)": _preset(vehicle="logistik", setup_time=SETUP_TIME_MAX),
}
# Werte in PRESET_HELP nach der Messreihe (sb_evaluation.run_config) final eingetragen.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": f"{DEFAULT_N} Aufträge auf {DEFAULT_M} Maschinen: Shifting Bottleneck misst sich gegen MWKR (Stück 8) und gegen sich selbst ohne Reoptimierung.",
    "Kleine Instanz (CP-SAT sichtbar)": f"{EXACT_MAX_N} Aufträge: hier löst CP-SAT (OR-Tools) das Problem exakt mit.",
    "Große Instanz (Skalierung)": f"{N_MAX} Aufträge: Shifting Bottleneck bleibt praktikabel, eine exakte Lösung wäre bei dieser Größe aussichtslos.",
    "Werkstatt/Logistik-Vehikel": "Dieselben Aufträge, aber in Familien mit Rüstzeit beim Wechsel je Maschine - die Teilprobleme werden dadurch selbst zu kleinen Reihenfolgeproblemen.",
    "Hohe Rüstlast (Werkstatt)": f"Rüstzeit {SETUP_TIME_MAX} Minuten je Familienwechsel: der Vorsprung von Shifting Bottleneck vor MWKR wächst tendenziell.",
}
