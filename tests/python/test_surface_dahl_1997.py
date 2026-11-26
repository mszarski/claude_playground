"""`extinction: dahl_1997` for the bubble-layer surface model (docs/tasks/SST_GAPS_JACKSON.md G-7;
issues sonar-64sm, sonar-9le). Added opt-in; the default since A1 (docs/references/CONFORMANCE.md
Sec. 1.1). `apl_1994`, the default before, stays selectable by name and its values are held
bit-identical below.

  [D97] P. H. Dahl, W. J. Plant, B. Nuetzel, A. Schmidt, H. Herwig and E. A. Terray, "Simultaneous
        acoustic and microwave backscattering from the sea surface", J. Acoust. Soc. Am. 101 (5),
        2583-2595 (1997). Read on the page images by docs/references/surface_papers_mcdaniel_dahl.md
        (the note); the formulas and numbers below are typed from the note, with its section.

The law: Table II (p. 2591), log10 beta_I = a + b log10 U10, fitted at 17 degrees grazing in the
North Sea (data 0-13 m/s, the fit dashed below 2.5 m/s as an extrapolation), at 26.5, 50 and
70 kHz. Between those frequencies log10 beta_I is interpolated linearly in log10 f (ours);
outside 26.5-70 kHz it is scaled as f^0.8, the trend [D97] state (p. 2592), with a warning
(ours: nothing in the paper is outside that band).

Values read from figures (Urick-Hoover in McDaniel 1993, Fig. 18) carry the note's +-1.5 dB and
bound nothing tightly: those tests are characterisations of where we stand, not validations.
"""

import math
import warnings

import numpy as np
import pytest

from gpusonar.sim import surface

TABLE_II = {26.5e3: (-6.16, 5.50), 50e3: (-6.70, 6.40), 70e3: (-5.75, 5.35)}   # [D97] Table II, p. 2591
DELTA_R = 0.0136
G17 = math.radians(17.0)


def table_ii(u, f):
    a, b = TABLE_II[f]
    return 10.0 ** (a + b * math.log10(u))


def beta(u, f, name="dahl_1997"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return surface.bubble_layer_optical_depth(u, f, extinction=name)


def surface_db(g, u, f, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return surface.bubble_layer_surface_db(g, u, f, extinction=name)


def test_table_ii_at_its_three_frequencies():
    """The option is [D97] Table II exactly where the table is printed (2.5 to 13 m/s, no warning there)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for f in TABLE_II:
            for u in (2.5, 3.0, 5.0, 8.0, 10.0, 13.0):
                assert surface.bubble_layer_optical_depth(u, f, extinction="dahl_1997") == pytest.approx(table_ii(u, f), rel=1e-12)


def test_the_notes_tabulated_values_and_our_laws_against_them():
    """[the note Sec. 3.2]: beta_I of Table II, and our two older laws over it in dB (positive: ours
    larger), at 2-15 m/s. Reproduced to the printed figures (2 significant figures; 0.1 dB)."""
    winds = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
    note = {26.5e3: ((3.1e-5, 2.9e-4, 1.4e-3, 4.8e-3, 1.3e-2, 6.4e-2, 0.22, 0.60, 2.0),
                     (+10.1, +5.7, +4.0, +3.9, +4.7, -0.2, -4.0, -7.1, -10.9),
                     (+17.5, +13.1, +10.0, +7.6, +5.6, +3.0, +2.2, +1.2, -0.2)),
            70e3: ((7.3e-5, 6.3e-4, 3.0e-3, 9.8e-3, 2.6e-2, 0.12, 0.40, 1.06, 3.5),
                   (+10.1, +5.9, +4.4, +4.4, +5.4, +0.7, -3.0, -6.0, -9.7),
                   (+19.5, +15.4, +12.5, +10.2, +8.3, +6.0, +5.5, +5.0, +4.2))}
    for f, (values, apl, hn) in note.items():
        for u, v, a, h in zip(winds, values, apl, hn):
            d = beta(u, f)
            assert d == pytest.approx(v, rel=0.05 if v < 1.0 else 0.03), (f, u)
            assert 10 * math.log10(beta(u, f, "apl_1994") / d) == pytest.approx(a, abs=0.1), (f, u)
            assert 10 * math.log10(beta(u, f, "hall_novarini") / d) == pytest.approx(h, abs=0.1), (f, u)
    for u, a in ((2.0, 15.2), (3.0, 9.1), (5.0, 5.3), (8.0, -0.6), (15.0, -13.8)):      # 50 kHz, the note's text
        assert 10 * math.log10(beta(u, 50e3, "apl_1994") / beta(u, 50e3)) == pytest.approx(a, abs=0.1), u


def test_between_and_outside_the_three_frequencies():
    """Between the table's frequencies: log10 beta_I linear in log10 f (ours), continuous at each.
    Outside 26.5-70 kHz: the nearest fit scaled as f^0.8 ([D97] p. 2592's trend; using it outside
    their band is ours), with a warning. Below 2.5 m/s (the dashed part of [D97]'s fit) and above
    13 m/s (beyond their data) a warning too; the value is still the fit's."""
    for u in (3.0, 6.0, 10.0):
        for lo, hi in ((26.5e3, 50e3), (50e3, 70e3)):
            for frac in (0.25, 0.5, 0.9):
                f = lo * (hi / lo) ** frac
                want = 10 ** ((1 - frac) * math.log10(table_ii(u, lo)) + frac * math.log10(table_ii(u, hi)))
                assert beta(u, f) == pytest.approx(want, rel=1e-12)
        for f in (26.5e3, 50e3, 70e3):
            assert beta(u, f * (1 + 1e-9)) == pytest.approx(table_ii(u, f), rel=1e-6)
            assert beta(u, f * (1 - 1e-9)) == pytest.approx(table_ii(u, f), rel=1e-6)
        assert beta(u, 12e3) == pytest.approx(table_ii(u, 26.5e3) * (12 / 26.5) ** 0.8, rel=1e-12)
        assert beta(u, 120e3) == pytest.approx(table_ii(u, 70e3) * (120 / 70) ** 0.8, rel=1e-12)
    with pytest.warns(UserWarning, match="26.5-70 kHz"):
        surface.bubble_layer_optical_depth(5.0, 120e3, extinction="dahl_1997")
    with pytest.warns(UserWarning, match="26.5-70 kHz"):
        surface.bubble_layer_optical_depth(5.0, 12e3, extinction="dahl_1997")
    with pytest.warns(UserWarning, match="below 2.5 m/s"):
        surface.bubble_layer_optical_depth(2.0, 50e3, extinction="dahl_1997")
    with pytest.warns(UserWarning, match="above 13 m/s"):
        surface.bubble_layer_optical_depth(15.0, 50e3, extinction="dahl_1997")
    assert beta(0.0, 50e3) == 0.0                         # no wind, no bubbles (the power law at U = 0)


def test_eq17_inversion_round_trip():
    """[D97] obtained Table II by inverting eq. (17) (Newton-Raphson) for beta_I from the measured
    level at 17 degrees. Inverting our model's bubble term the same way returns the optical depth it
    was given, on the rising branch of eq. (17) (beta_I / sin t below 0.6, the overshoot's peak)."""
    def eq17(b_i, f):
        b = b_i / math.sin(G17)
        delta = 2.55e-3 * f ** (1 / 3)
        return math.sin(G17) / (8 * math.pi) * DELTA_R / delta * (1 + 8 * b * math.exp(-2 * b) - math.exp(-4 * b))

    for f in TABLE_II:
        for u in (2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            want = beta(u, f)
            if want / math.sin(G17) > 0.6:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                level = float(surface.bubble_backscatter(G17, u, f, extinction="dahl_1997"))
            x = 1e-6
            for _ in range(60):                           # Newton-Raphson with a numerical derivative
                h = 1e-7 * x
                slope = (eq17(x + h, f) - eq17(x - h, f)) / (2 * h)
                x = max(x - (eq17(x, f) - level) / slope, 0.5 * x)
            assert x == pytest.approx(want, rel=1e-8), (f, u)


def test_wind_exponent_at_17_degrees_against_d97():
    """[D97] p. 2591: measured sigma_0 ~ U^n at 17 degrees over 2-7 m/s, n about 5.5, 6.4, 5.3 at 26.5,
    50, 70 kHz (printed; "about"). Fitted here the same way for every extinction law: least squares
    of 10 log10 sigma (bubble + facet) against log10 U on 51 winds uniform in 2-7 m/s:
        dahl_1997      5.37 / 6.20 / 5.10   (the note: 5.37 / 6.19 / 5.10)
        apl_1994       4.25 / 4.06 / 3.91   (the note: 4.13 / 3.94 / 3.80; apl_1994 changes form at
                                             6 m/s, so its exponent depends on how the winds are
                                             sampled: 3.4 to 4.25 for the end points, integer winds,
                                             log- or linearly spaced winds)
        hall_novarini  2.63 / 2.12 / 1.72
    The published law brings the exponent to within 0.2 of the measurement; the older laws are 1.2 to
    3.6 below it. The measured exponents are rounded, single-site values: the 0.3 bound is no tighter
    than they are."""
    u = np.linspace(2.0, 7.0, 51)
    measured = {26.5e3: 5.5, 50e3: 6.4, 70e3: 5.3}
    ours = {"dahl_1997": (5.37, 6.20, 5.10), "apl_1994": (4.25, 4.06, 3.91), "hall_novarini": (2.63, 2.12, 1.72)}
    for name, want in ours.items():
        for f, n_want in zip(TABLE_II, want):
            s = np.array([float(surface_db(G17, x, f, name)) for x in u])
            n = float(np.polyfit(np.log10(u), s / 10.0, 1)[0])
            assert n == pytest.approx(n_want, abs=0.02), (name, f, n)
            if name == "dahl_1997":
                assert abs(n - measured[f]) < 0.3, (f, n)
            else:
                assert measured[f] - n > 1.0, (name, f, n)


def test_urick_hoover_readings_at_60_khz():
    """Characterisation only (figure readings, +-1.5 dB; +-2 dB at the curve ends): Urick and Hoover (1956),
    60 kHz, open ocean, as McDaniel (1993) Fig. 18 shows them (the note Sec. 2.7). Ours (bubble + facet)
    minus the reading, dB, at the grazing angles listed:
        wind 9.0 m/s, 15/30/45/60 deg:  apl_1994 +1.2 / +3.4 / +2.6 / +0.6    dahl_1997 +0.8 / +4.0 / +3.6 / +1.4
             6.4                        +9.0 / +9.8 / +8.0 / +2.9              +6.5 / +6.5 / +4.4 / -0.4
             3.9                       +12.2 / +10.5 / +7.1 / +0.1             +7.0 / +5.2 / +1.7 / -5.2
             2.6, 32/45/60 deg          +9.3 / +5.8 / +2.3                     +0.5 / -3.0 / -6.5
    With dahl_1997 every point is within +8 dB (apl_1994 reaches +12.2). The note's own "with [D97]
    beta_I" column (+7.9 / +6.2 / +2.7 / -4.2 at 3.9 m/s; +2.3 / -1.2 / -4.7 at 2.6 m/s) scaled the 70 kHz
    fit by f^0.8 instead of interpolating between 50 and 70 kHz: up to 1.8 dB apart at 2.6 m/s.
    The facet term attenuated by the layer (A2) moves only the 60 degree points: before A2
    (`facet_attenuation=False`) 9.0 m/s +1.1 / +1.9 and 6.4 m/s +3.0 / -0.3 (apl_1994 / dahl_1997)."""
    readings = {9.0: {15: -26.0, 30: -26.0, 45: -24.5, 60: -21.0}, 6.4: {15: -34.3, 30: -33.8, 45: -31.5, 60: -26.0},
                3.9: {15: -46.3, 30: -44.5, 45: -41.0, 60: -34.0}, 2.6: {32: -50.0, 45: -46.5, 60: -43.0}}
    ours = {9.0: (0.8, 4.0, 3.6, 1.4), 6.4: (6.5, 6.5, 4.4, -0.37), 3.9: (7.0, 5.2, 1.7, -5.2), 2.6: (0.5, -3.0, -6.5)}
    apl = {9.0: (1.2, 3.4, 2.6, 0.63), 6.4: (9.0, 9.8, 8.0, 2.92), 3.9: (12.2, 10.5, 7.1, 0.1), 2.6: (9.3, 5.8, 2.3)}
    for u, points in readings.items():
        g = np.radians(list(points))
        read = np.array(list(points.values()))
        d = surface_db(g, u, 60e3, "dahl_1997") - read
        np.testing.assert_allclose(d, ours[u], atol=0.06)
        assert np.all(d < 8.0)
        np.testing.assert_allclose(surface_db(g, u, 60e3, "apl_1994") - read, apl[u], atol=0.06)


# Values computed from the tree before `dahl_1997` was added (commit f2c34bc), as float.hex:
# bubble_layer_surface_db at 5 / 17 / 30 / 45 / 60 / 85 degrees, then beta_I of apl_1994 and hall_novarini.
BEFORE = {
    (3.0, 68e3): (("-0x1.3417d249e9789p+5", "-0x1.32c38b560727ep+5", "-0x1.328761ae67700p+5", "-0x1.326e8e148f6cfp+5",
                   "-0x1.325d8450e40c9p+5", "0x1.2800f23077a84p+2"), "0x1.388fda1b47dc5p-9", "0x1.5ea45ba2764c8p-6"),
    (5.0, 30e3): (("-0x1.ef8fc983567cfp+4", "-0x1.e134c55e33520p+4", "-0x1.dea3780a8341ap+4", "-0x1.dd9392ec542e2p+4",
                   "-0x1.da4d692208b04p+4", "0x1.acd577a69f666p+1"), "0x1.ada37d58479a8p-7", "0x1.0c43efb5e9d62p-5"),
    (10.0, 50e3): (("-0x1.f684e3bf019f3p+4", "-0x1.80a2dabe3aeefp+4", "-0x1.65cea3886a0d9p+4", "-0x1.5a4efc3952505p+4",
                    "-0x1.2e0f6fb1347d6p+4", "0x1.258739cfb33b6p+0"), "0x1.32e8e5c75a1b2p-3", "0x1.d8df383c677c3p-1"),
    (15.0, 120e3): (("-0x1.12140920dd255p+5", "-0x1.bf1dbbffcbccbp+4", "-0x1.7f441b368e601p+4", "-0x1.5b47eceb08971p+4",
                     "-0x1.ce7ef254b65c6p+3", "-0x1.42a1777b0ee9cp-2"), "0x1.3134d1f717b96p-1", "0x1.f3a354abab58fp+3"),
}


def test_apl_1994_path_is_bit_identical_when_named():
    """`apl_1994` (the default until A1) and `hall_novarini`, when named, give exactly the values the
    default and `hall_novarini` gave before `dahl_1997` was added, bit for bit, through
    `bubble_layer_surface_db` (with `facet_attenuation=False`, the model before A2) and
    `bubble_layer_optical_depth`. (Until A1 this test also held the unnamed default to these values;
    the default is now `dahl_1997`: `test_dahl_1997_is_the_default_extinction`.)"""
    g = np.radians([5.0, 17.0, 30.0, 45.0, 60.0, 85.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (u, f), (levels, apl, hn) in BEFORE.items():
            assert [float(x).hex() for x in surface.bubble_layer_surface_db(g, u, f, extinction="apl_1994",
                                                                              facet_attenuation=False)] == list(levels)
            assert float(surface.bubble_layer_optical_depth(u, f, extinction="apl_1994")).hex() == apl
            assert float(surface.bubble_layer_optical_depth(u, f, extinction="hall_novarini")).hex() == hn


def test_unknown_extinction_is_refused_naming_the_three():
    with pytest.raises(ValueError, match="unknown extinction 'dahl'.*apl_1994.*dahl_1997.*hall_novarini"):
        surface.bubble_layer_optical_depth(5.0, 50e3, extinction="dahl")


def test_scenario_item_takes_dahl_1997():
    """`{model: bubble_layer_surface, extinction: dahl_1997}` reaches the law through `run_scenario`
    (its frequency-range warning is raised: the scenario is at 300 kHz)."""
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("point_grid_fls_300k")
    cfg["n_pings"] = 1
    cfg["reverberation"] = [{"model": "bubble_layer_surface", "boundary_z": 0.0, "wind_speed": 8.0, "extinction": "dahl_1997"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = run_scenario(cfg).records
    assert len(records) == 1 and np.all(np.isfinite(records[0].samples.array))
    assert any("dahl_1997" in str(w.message) and "26.5-70 kHz" in str(w.message) for w in caught)


def test_dahl_1997_is_the_default_extinction():
    """A1 (docs/references/CONFORMANCE.md Sec. 1.1): `dahl_1997` is the default extinction of the
    bubble-layer surface model, in every function that takes the keyword and in the scenario item
    (`{model: bubble_layer_surface}` without `extinction`). The printed [D97] Table II fit wins over
    `apl_1994` (a secondary's loss law, 5.2 dB above [D99]'s beta_I <= 0.0012 at 30 kHz, 4 m/s, where
    `dahl_1997` is 0.7 dB above: [NB] Sec. 3.2). `apl_1994` stays selectable by name."""
    import inspect
    for fn in (surface.bubble_layer_optical_depth, surface.bubble_backscatter, surface.bubble_layer_surface_db,
               surface.bubble_layer_surface_strength):
        assert inspect.signature(fn).parameters["extinction"].default == "dahl_1997", fn.__name__
    g = np.radians([5.0, 17.0, 45.0, 80.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # inside both stated ranges: no warning
        for u, f in ((3.0, 68e3), (5.0, 30e3), (10.0, 50e3)):
            assert surface.bubble_layer_optical_depth(u, f) == surface.dahl_1997_optical_depth(u, f)
            assert np.array_equal(surface.bubble_layer_surface_db(g, u, f), surface.bubble_layer_surface_db(g, u, f, extinction="dahl_1997"))
            assert not np.array_equal(surface.bubble_layer_surface_db(g, u, f), surface.bubble_layer_surface_db(g, u, f, extinction="apl_1994"))
    # [D99] p. 2161: beta_I <= 0.0012 at 30 kHz, about 4 m/s (an upper bound): the default is 0.7 dB above it,
    # apl_1994 5.2 dB ([NB] Sec. 3.2).
    assert 10 * math.log10(surface.bubble_layer_optical_depth(4.0, 30e3) / 0.0012) == pytest.approx(0.7, abs=0.05)
    assert 10 * math.log10(surface.bubble_layer_optical_depth(4.0, 30e3, extinction="apl_1994") / 0.0012) == pytest.approx(5.2, abs=0.05)


def test_scenario_item_defaults_to_dahl_1997():
    """The scenario item without `extinction` is the item with `extinction: dahl_1997`, bit for bit,
    and not the one with `extinction: apl_1994` (A1)."""
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("point_grid_fls_300k")
    cfg["n_pings"] = 1
    item = {"model": "bubble_layer_surface", "boundary_z": 0.0, "wind_speed": 8.0}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                      # 300 kHz is outside both stated ranges
        for name in (None, "dahl_1997", "apl_1994"):
            cfg["reverberation"] = [dict(item) if name is None else dict(item, extinction=name)]
            out[name] = run_scenario(cfg).records[0].samples.array.copy()
    assert np.array_equal(out[None], out["dahl_1997"])
    assert not np.array_equal(out[None], out["apl_1994"])
