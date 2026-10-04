# Shifting Bottleneck – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-shifting-bottleneck-demo.streamlit.app/)**

Neuntes Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning", das erste SOTA-Stück: dasselbe Job-Shop-Modell wie Stück 8
(`job-shop-demo`), aber ein anderes VERFAHREN - statt EINER Prioritätsregel für alle Maschinen zugleich
sequenziert das **Shifting-Bottleneck-Verfahren** (Adams, Balas & Zawack 1988) die Maschinen NACHEINANDER, immer
die engpassträchtigste zuerst, und **reoptimiert** bereits sequenzierte Maschinen danach.

**Einordnung in die Linie:**
```
                                                             ┌─ Shifting Bottleneck (dieses Stück)
Job Shop (Giffler-Thompson + MWKR, Stück 8 - Konvergenzpunkt) ┤
                                                             └─ Job-Shop-Tabu-Search (nächstes SOTA-Stück)
```
Giffler-Thompson (Stück 8) wendet EINE Prioritätsregel auf ALLE Maschinen gleichzeitig an. Shifting Bottleneck
geht anders vor: für jede noch offene Maschine wird ein eigenes `1|rⱼ|Lmax`-Teilproblem gelöst (Freigabezeiten =
Heads, Fälligkeiten = Horizont − Tails), die Maschine mit dem GRÖSSTEN gelösten Lmax ist der Engpass und wird
fest sequenziert - danach werden alle bereits sequenzierten Maschinen mit den aktualisierten Heads/Tails erneut
gelöst ("Reoptimierung").

Ergebnis in Kürze: bei 10 Aufträgen auf 4 Maschinen (Standard-Seed) liegt Cmax bei der Fassung ohne Reoptimierung **18,9 %**
und bei MWKR (Stück 8) **16,1 %** über dem von **Shifting Bottleneck**. Über die feste Messreihe (5 Instanzen)
liegt Cmax ohne Reoptimierung im Mittel **11,1 %** und bei MWKR im Mittel **1,7 %** über dem von Shifting Bottleneck. **Der
ehrliche Befund dieses Stücks**: WEDER die Reoptimierung NOCH das Verfahren insgesamt sind bewiesen monoton
besser - bei n=3 (Seed 100000) schneidet die Reoptimierung minimal schlechter ab als ganz ohne, bei n=10 (Seed
100000) verliert Shifting Bottleneck sogar gegen MWKR. Ein zweiter, tieferer Fund beim Bau: das
`1|rⱼ|Lmax`-Teilproblem respektiert nur Freigabezeiten (eine Relaxation der vollen Graphstruktur) - eine für
sich genommen optimale Reihenfolge auf einer neuen Maschine kann, kombiniert mit bereits fixierten Maschinen,
über den Auftragsvorrang einen ZYKLUS schließen (ein dokumentiertes Risiko der Head/Tail-Relaxation). Die Demo
erkennt das und weicht beweisbar zyklenfrei aus (aufsteigender Head) - siehe Verifikation/Tests.

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004) |
|---|---|
| Standardfall (10 Aufträge, 4 Maschinen, Seed 60) | ✅ Cmax liegt ohne Reoptimierung **18,9 %**, bei MWKR **16,1 %** über dem von Shifting Bottleneck |
| **Reoptimierung im Mittel** | ✅ Cmax ohne Reoptimierung liegt im Mittel **11,1 %** über dem mit Reoptimierung |
| **Shifting Bottleneck gegen MWKR (Stück 8) im Mittel** | ✅ Cmax von MWKR liegt im Mittel **1,7 %** über dem von Shifting Bottleneck |
| **Reoptimierung auf JEDER Instanz besser?** | ❌ Nein - n=3, Seed 100000: minimal schlechter als ohne |
| **Shifting Bottleneck auf JEDER Instanz besser als MWKR?** | ❌ Nein - n=10, Seed 100000: schlechter als MWKR |
| **Zyklen-Risiko der Teilproblem-Relaxation** | ⚠️ Real (n=20, Seed 100003 ohne Reoptimierung) - erkannt und behoben |

## Was die Demo zeigt

1. **Shifting Bottleneck in Aktion** (Schritt-Slider): **Aufträge** (gestapelter Balken je Auftrag in eigener
   Maschinenreihenfolge) → **Engpässe nacheinander** (Regler "bereits sequenzierte Maschinen", Gantt mit einer
   Zeile JE MASCHINE, Farbe nach AUFTRAG, Bildunterschrift zeigt die TATSÄCHLICH gewählte Engpass-Reihenfolge) →
   **Ergebnis** (Fertigstellung je Maschine, Shifting Bottleneck gegen MWKR).
2. **Was Reoptimierung und das Verfahren bringen:** Shifting Bottleneck (Cmax), Abstand ohne Reoptimierung,
   Abstand zu MWKR (Stück 8), CP-SAT-Gegenprobe (n ≤ 8, mit Beweis-Status).
3. **📐 Sweep** über die Anzahl der Aufträge ODER Maschinen.
4. **🔬 Experimente auf Abruf:** hilft die Reoptimierung wirklich, und wie oft (der Beweis-Check dieses Stücks -
   ehrlich gemessen, keine 100-%-Garantie); Rechenzeit CP-SAT (volles Modell) gegen Shifting Bottleneck (viele
   kleine Teilprobleme); Rüstzeit-Härtetest.
5. **🚧 Grenzen:** Tabelle mit Verweis auf das nächste SOTA-Stück (Job-Shop-Tabu-Search).

Regler: Aufträge (2–20), **Maschinen** (2–6), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich
Rüstzeit und Anzahl Familien), Seed der Instanz (+ 🎲).

## Die zwei Vehikel (gelten für die ganze Linie)

- **Neutral** (`sb_scenario.py`): dasselbe Instanzmodell wie Stück 8 - $n$ Aufträge, jeder mit einer zufälligen
  Permutation der $m$ Maschinen als eigener Reihenfolge, Bearbeitungszeiten $\sim U(1, 100)$ je Operation.
- **Werkstatt/Logistik** (`sb_scenario_logistik.py`): dieselbe Instanz, aber jeder Auftrag gehört zu einer
  Familie; ein Familienwechsel kostet eine feste Rüstzeit JE MASCHINE. Für Shifting Bottleneck macht das jedes
  Ein-Maschinen-Teilproblem selbst zu einem kleinen Reihenfolgeproblem (Kreis-Modell statt einfachem
  Intervall+NoOverlap) - Rüstzeit 0 kollabiert strukturell exakt zum neutralen Vehikel (per Test belegt).

## Modell und Verfahren

- **Instanz** (`sb_scenario.py`, `sb_scenario_logistik.py`): Routing, Bearbeitungszeiten, Familien und
  Rüstzeit-Matrix - identisch zu Stück 8, eigener Code für dieses Repo.
- **Heads/Tails** (`sb_algorithm.compute_heads`/`compute_tails`): längster Weg über Auftragsvorrang PLUS bereits
  fest sequenzierte Maschinen (inklusive Rüstzeit bei Familienwechsel) - liefert Freigabezeit und Fälligkeit je
  Operation, nur mit den bisher sequenzierten Maschinen.
- **Teilproblem je Maschine** (`sb_algorithm.solve_1r_lmax`): `1|rⱼ|Lmax` (Neutral, einfaches
  Intervall+NoOverlap-Modell) bzw. `1|rⱼ,sⱼₖ|Lmax` (Werkstatt/Logistik, Kreis-Modell mit sequenzabhängiger
  Rüstzeit) über CP-SAT - bewusste Vereinfachung statt Carliers (1982) spezialisiertem Branch-and-Bound (mit dem
  Nutzer abgestimmt).
- **Zyklen-Schutz** (`sb_algorithm._is_acyclic`/`_assign_machine_sequence`): nach jeder Teilproblem-Lösung wird
  geprüft, ob die kombinierte Reihenfolge (Auftragsvorrang + alle bereits fixierten Maschinen) azyklisch bleibt -
  falls nicht, beweisbar sicherer Rückfall (aufsteigender Head).
- **Shifting-Bottleneck-Hauptschleife** (`sb_algorithm.shifting_bottleneck`): Engpasswahl (größtes Lmax) +
  optionale Reoptimierung bereits sequenzierter Maschinen.
- **CP-SAT** (`sb_algorithm.solve_exact`): dasselbe Kreis-Modell wie Stück 8, hier als exakte Gegenprobe auf das
  VOLLE Problem. Gegen unabhängige Brute-Force-Vollaufzählung verifiziert.
- **Auswertung** (`sb_evaluation.py`): Kennzahlen, Sweep, Reoptimierungs-Check, Timing-Messreihe,
  Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Annahme: "Mit Reoptimierung ist das Ergebnis nie schlechter als ohne"** - **klar widerlegt** durch eine
  erste kleine Stichprobe (12 Instanzen, 3 Seeds), die zufällig KEIN Gegenbeispiel enthielt. Eine breitere
  Messreihe (5 feste Seeds × mehrere n) fand echte, reproduzierbare Gegenbeispiele (z. B. n=3, m=4, Seed 100000:
  382 statt 377). Grund: die Reoptimierung verändert die Heads/Tails ALLER noch offenen Maschinen und damit die
  ENGPASS-REIHENFOLGE selbst - kein rein lokaler, monoton verbessernder Schritt auf einer festen Struktur. Die
  Lehre aus Stück 7/8 ("ein Verfahren, das im Mittel hilft, ist keine bewiesene Garantie auf jeder Instanz") gilt
  hier NOCHMAL, diesmal für ein ganzes Verfahren statt für eine einzelne Prioritätsregel.
- **Ein echter, ernster Bug beim Bau: unentdeckte Zyklen.** Bei n=20, m=4, Seed 100003 (ohne Reoptimierung)
  lieferte die ursprüngliche Implementierung Cmax = 42176 - physikalisch unmöglich (Bearbeitungszeiten-Summe nur
  3854). Ursache: das `1|rⱼ|Lmax`-Teilproblem respektiert nur Freigabezeiten (eine Relaxation), nicht die volle
  Graphstruktur - CP-SAT kann für die NEUE Maschine eine lokal optimale Reihenfolge wählen, die zusammen mit
  bereits fixierten Maschinen über den Auftragsvorrang einen ZYKLUS schließt (per `networkx` nachgewiesen: sechs
  Operationen über drei Maschinen, exakt der klassische "Zyklus im disjunktiven Graphen"). Ohne Erkennung lief
  die Fixpunkt-Iteration von `compute_heads` einfach unkonvergiert bis zur Rundenobergrenze durch und lieferte
  stillschweigend Unsinn. **Fix**: nach jeder Teilproblem-Lösung eine Kahn-Azyklizitätsprüfung; im (seltenen)
  Zyklus-Fall ein beweisbar sicherer Rückfall - Operationen der neuen Maschine nach AUFSTEIGENDEM Head sortiert
  (eine Kante von kleinerem zu größerem-oder-gleichem Head kann in einem bereits azyklischen Graphen unmöglich
  einen Rückwärtspfad schließen). Permanente Regressionstests in `tests/test_algorithm.py`.
- **Nichtdeterminismus durch parallele CP-SAT-Suche.** Dieselbe Instanz lieferte je nach `num_search_workers`
  (4 vs. 8) unterschiedliche, beide gültig optimale Lösungen für ein Teilproblem - weil mehrere gleichwertige
  Reihenfolgen (Lmax-Bindungen) existieren und verschiedene Worker-Konfigurationen unterschiedlich tiebreaken.
  Das hätte bedeutet: derselbe Permalink zeigt lokal ein anderes Ergebnis als auf Streamlit Cloud (andere
  Kernzahl). **Fix**: `solve_1r_lmax` löst IMMER mit genau 1 Worker (die Teilprobleme sind winzig, das kostet
  keine spürbare Zeit) - nur die einmalige CP-SAT-Gegenprobe auf das volle Modell nutzt weiterhin mehrere Worker.
- **Der Rüstzeit-0-Fall braucht denselben Modellpfad wie ganz ohne Familien.** Ursprünglich nutzte
  `solve_1r_lmax` bei `family is not None` immer das Kreis-Modell (auch bei Rüstzeit 0) - das lieferte bei
  Bindungen ein ANDERES Tiebreak-Ergebnis als der einfache Intervall-Pfad und brach damit den erwarteten
  "Rüstzeit 0 = neutrales Vehikel"-Kollaps. Fix: bei `setup.max() == 0` bewusst denselben einfachen Pfad nehmen.
- **Das 1\|rⱼ\|Lmax-Teilproblem wird über CP-SAT statt Carliers (1982) spezialisiertem Branch-and-Bound-Verfahren
  gelöst** - eine bewusste, mit dem Nutzer abgestimmte Vereinfachung, für die Größenordnungen dieser Demo ohne
  praktischen Unterschied.
- **Synthetische Instanzen:** wie Stück 8 - Bearbeitungszeiten gleichverteilt, jeder Auftrag besucht jede
  Maschine genau einmal.

## Verifikation

- **CP-SAT gegen unabhängige Brute-Force-Vollaufzählung** (mit und ohne Rüstzeiten): für n ≤ 4, m ≤ 3 über
  mehrere Seeds stimmt das Kreis-Modell exakt mit einer unabhängigen Vollaufzählung überein.
- **Shifting Bottleneck gegen CP-SAT-Optimum**: für n = 2 bis 4 (mehrere Seeds, mit und ohne Rüstzeit) bleibt
  Shifting Bottleneck innerhalb einer festen Toleranz zum bewiesenen Optimum.
- **Heads/Tails-Eigenschaften**: Auftragsvorrang und fixierte Maschinenreihenfolgen werden respektiert (mit
  Handrechnung für die Rüstzeit-Variante).
- **Zyklen-Schutz**: von Hand konstruiertes Gegenbeispiel (`_is_acyclic` erkennt den Zyklus), Regressionstest für
  den echten Fund (n=20, m=4, Seed 100003).
- **Struktureller Konsistenz-Test**: Shifting Bottlenecks Zeitplan bei Rüstzeit 0 ist identisch mit dem neutralen
  Vehikel.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Reoptimierungs-Check, Rüstzeit-
  Härtetest; positive **und** negative Aussagen inklusive der beiden Gegenbeispiele); alle 5 Presets geprüft;
  AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt auf beiden Vehikeln, Würfel-Knopf,
  Permalink-Grenzen inkl. ungültigem Vehikel, Extremwerte, Experimente auf Abruf, Footer, korrekt formatierte
  negative Prozent-Abstände); eigener Test für die ausblendbaren Regler.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweep, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `sb_algorithm.py` | Heads/Tails, Teilproblem-Löser, Zyklen-Schutz, Shifting-Bottleneck-Hauptschleife, Giffler-Thompson/MWKR (Vergleichsbasis), CP-SAT |
| `sb_scenario.py` | Vehikel Neutral |
| `sb_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `sb_constants.py` | Konstanten, Presets |
| `sb_evaluation.py` | Kennzahlen, Sweep, Reoptimierungs-Check, Timing-Messreihe, Rüstzeit-Härtetest |
| `sb_presets.py`, `sb_visualization.py` | Permalink/Presets (inkl. `seed_widget`/`KEPT` für ausblendbare Regler), Plotly-Figuren (ein Trace je Maschine, achsengesperrt) |
| `tests/` | CP-SAT gegen Vollaufzählung, Shifting Bottleneck gegen CP-SAT, Zyklen-Schutz, Szenario und Auswertung, Aussagen der App, Presets, versteckter Widget-Zustand, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Scheduling-Theorie: SPT bis RCPSP](https://sebastianhanisch.net/konzepte-klassische-scheduling-theorie.html).
