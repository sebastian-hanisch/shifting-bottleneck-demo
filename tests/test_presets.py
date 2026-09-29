"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten."""

import sb_constants as C
import sb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 5
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX
        assert C.M_MIN <= preset["m"] <= C.M_MAX
        assert preset["vehicle"] in C.VEHICLE_LABELS
        assert C.SETUP_TIME_MIN <= preset["setup_time"] <= C.SETUP_TIME_MAX
        assert C.N_FAMILIES_MIN <= preset["n_families"] <= C.N_FAMILIES_MAX
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert p["n"] == C.DEFAULT_N and p["m"] == C.DEFAULT_M and p["seed"] == C.DEFAULT_SEED and p["vehicle"] == C.DEFAULT_VEHICLE


def test_small_preset_is_within_the_exact_limit():
    p = C.PRESETS["Kleine Instanz (CP-SAT sichtbar)"]
    assert p["n"] <= C.EXACT_MAX_N


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.bounds("m_slider") == (C.M_MIN, C.M_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
