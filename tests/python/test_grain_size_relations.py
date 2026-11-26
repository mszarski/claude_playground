"""The TR 9407 grain-size relations as Jackson and Richardson (2007) print them, eqs. (11.1)-(11.3),
pp. 313-314 [J&R], as an opt-in method of `bottom.sediment_from_grain_size` (docs/tasks/SST_GAPS_JACKSON.md
G-6; notes docs/references/jackson_richardson_2007/01 Sec. 2.4, 3.2 [N1] and 03 Sec. 2.5, 3.2, 6 [N3]).

Two printed constants are taken as misprints, as the notes argue: 1.0019, not 1.0091, in the last piece
of (11.2) (continuity with the cubic at 5.3 phi, and the tabulated rows 5.5-9 phi), and 0.4556, not
0.4566, in the first piece of (11.3) (the next piece starts at 0.4556, and the rows at -1 and -0.5 phi
follow it). The book calls the relations "ad hoc" model-fit expressions, not geoacoustic regressions
(p. 314). Today's linear interpolation of the half-phi table stays the default.
"""
import math

import numpy as np
import pytest

from gpusonar.sim import bottom

# The printed pieces, typed here again from [N1 2.4] / [N3 2.5] (independent of bottom.py).
K_PIECES = (  # (from Mz, to Mz, coefficients of Mz^2, Mz, 1), dB m^-1 kHz^-1, (11.3)
    (-1.0, 0.0, (0.0, 0.0, 0.4556)), (0.0, 2.6, (0.0, 0.0245, 0.4556)), (2.6, 4.5, (0.0, 0.1245, 0.1978)),
    (4.5, 6.0, (0.20098, -2.5228, 8.0399)), (6.0, 9.5, (0.0117, -0.2041, 0.9431)))


def k_piece(i, mz):
    a, b, c = K_PIECES[i][2]
    return a * mz * mz + b * mz + c


def normal_incidence_loss_db(par):
    """-20 log10 |V(90 deg)| (2.11) of the fluid half-space."""
    return -20.0 * math.log10(abs(bottom.reflection(0.5 * math.pi, par["rho"], par["nu"], par["delta"])))


def old_sediment_from_grain_size(phi):
    """`sediment_from_grain_size` exactly as it computed before the method was added."""
    table = np.array(bottom.GRAIN_SIZE_TABLE)
    out = {key: float(np.interp(phi, table[:, 0], table[:, i])) for i, key in ((1, "rho"), (2, "nu"), (3, "delta"))}
    return dict(out, **bottom.grain_size_scattering(phi))


def test_the_default_is_todays_interpolation_bit_for_bit():
    grid = np.round(np.linspace(-1.0, 9.0, 1001), 10)
    for phi in grid:
        phi = float(phi)
        old = old_sediment_from_grain_size(phi)
        assert bottom.sediment_from_grain_size(phi) == old, phi                       # dict equality: every float identical
        assert bottom.sediment_from_grain_size(phi, method="table_linear") == old, phi
    with pytest.raises(ValueError, match="unknown method 'relation'.*did you mean 'relations'"):
        bottom.sediment_from_grain_size(2.0, method="relation")
    with pytest.raises(ValueError, match="outside the range"):
        bottom.sediment_from_grain_size(9.5, method="relations")


# rho, nu, delta of `sediment_from_grain_size` written out (repr) by the code before this change (commit
# c8855ef) at the grain sizes where the two methods differ most; the full 0.01 phi grid was compared too.
BEFORE = {
    -0.73: (2.44286, 1.320638, 0.0168448),
    1.25: (1.9979999999999998, 1.20115, 0.016345),
    3.27: (1.30066, 1.067472, 0.0180738),
    5.3: (1.157, 0.99306, 0.009100000000000002),
    5.79: (1.149, 0.987804, 0.005078),
    6.36: (1.14828, 0.986436, 0.0032839999999999996),
    8.51: (1.14549, 0.981176, 0.0015535),
}


def test_the_default_matches_values_saved_from_the_tree_before_the_change():
    for phi, values in BEFORE.items():
        got = bottom.sediment_from_grain_size(phi)
        assert (got["rho"], got["nu"], got["delta"]) == values, phi                   # exact
        assert {k: got[k] for k in ("w2", "gamma2", "sigma2")} == bottom.grain_size_scattering(phi)


def test_the_relations_reproduce_every_tabulated_row():
    """[N1 3.2]: rho within 4.5e-4, nu within 4.9e-5, delta within 0.27 % (c_w 1528 m/s); held at the
    plan's 5e-4, 1e-4, 0.5 %. The tabulated points are where both methods meet."""
    for mz, rho, nu, delta in bottom.GRAIN_SIZE_TABLE:
        rel = bottom.apl_grain_size_relations(mz)
        assert rel["rho"] == pytest.approx(rho, abs=5e-4), mz
        assert rel["nu"] == pytest.approx(nu, abs=1e-4), mz
        assert rel["delta"] == pytest.approx(delta, rel=5e-3), mz
        par = bottom.sediment_from_grain_size(mz, method="relations")
        assert {k: par[k] for k in ("rho", "nu", "delta")} == {k: rel[k] for k in ("rho", "nu", "delta")}
        for key in ("w2", "gamma2", "sigma2"):                                        # the scattering parameters do not depend on the method
            assert par[key] == bottom.sediment_from_grain_size(mz)[key]


def test_delta_follows_from_k_by_2_6_with_the_tables_water():
    """delta_p = k nu_p c_w ln 10 / (40 000 pi) ((2.6) with alpha_p = k f_kHz); c_w = 1528 m/s reproduces
    0.01624 at 1.5 phi [N1 3.1]; another water speed scales delta with it."""
    rel = bottom.apl_grain_size_relations(1.5)
    assert rel["delta"] == pytest.approx(rel["k_db_per_m_khz"] * rel["nu"] * 1528.0 * math.log(10.0) / (40000.0 * math.pi), rel=1e-14)
    assert rel["delta"] == pytest.approx(0.01624, abs=5e-6)
    other = bottom.apl_grain_size_relations(1.5, water_sound_speed=1500.0)
    assert other["delta"] == pytest.approx(rel["delta"] * 1500.0 / 1528.0, rel=1e-14)
    assert (other["rho"], other["nu"]) == (rel["rho"], rel["nu"])


def test_density_and_speed_are_continuous_at_the_joins_and_1_0091_is_not_used():
    """(11.1), (11.2) at 1 and 5.3 phi: the pieces meet to 1e-4 (measured: rho 5.1e-5 and 5.1e-5, nu 6.9e-5
    and 1.4e-5). The printed 1.0091 would jump by 0.0072 at 5.3 phi (0.99621 against 0.98902) and miss
    every row from 5.5 to 9 phi by 0.0072."""
    eps = 1e-9
    for join in (1.0, 5.3):
        below, at = bottom.apl_grain_size_relations(join - eps), bottom.apl_grain_size_relations(join)
        assert at["rho"] == pytest.approx(below["rho"], abs=1e-4) and at["nu"] == pytest.approx(below["nu"], abs=1e-4), join
    assert bottom.apl_grain_size_relations(5.3)["nu"] == pytest.approx(0.98901, abs=1e-5)
    assert -0.0024324 * 5.3 + 1.0091 == pytest.approx(0.99621, abs=1e-5)                   # the printed piece, for the record
    assert bottom.apl_grain_size_relations(9.0)["nu"] == pytest.approx(0.9800, abs=1e-4)     # the table's clay; 1.0091 gives 0.9872


def test_the_printed_steps_of_11_3_contradiction_c2():
    """(11.3) as printed is not continuous. Evaluating the printed pieces at their joins settles the plan's
    contradiction C2: at 6.0 phi the two pieces give 0.1384 and 0.1397 dB/m/kHz (a step of 0.0013), as
    [N1 WP-A] says ("jumps 0.001-0.002"); [N3 2.5]'s "0.338 against 0.140" is an arithmetic slip. The
    other steps: 0 at 0 phi with 0.4556 (0.0010 with the printed 0.4566), +0.0022 at 2.6, -0.0009 at 4.5.
    Our relations keep the steps as printed (the rows follow the upper piece at each join), and C3: the
    first piece is 0.4556."""
    steps = {}
    for i in range(1, len(K_PIECES)):
        join = K_PIECES[i][0]
        steps[join] = k_piece(i, join) - k_piece(i - 1, join)
    assert steps[0.0] == pytest.approx(0.0, abs=1e-12)
    assert steps[2.6] == pytest.approx(0.0022, abs=1e-6)
    assert steps[4.5] == pytest.approx(-0.000905, abs=1e-6)
    assert steps[6.0] == pytest.approx(0.00132, abs=1e-6)
    assert (round(k_piece(3, 6.0), 4), round(k_piece(4, 6.0), 4)) == (0.1384, 0.1397)
    assert abs(0.4566 - k_piece(1, 0.0)) == pytest.approx(0.0010, abs=1e-9)
    for i, (lo, hi, _) in enumerate(K_PIECES):
        for mz in np.linspace(lo, min(hi, 9.0), 7)[:-1]:
            assert bottom.apl_grain_size_relations(mz)["k_db_per_m_khz"] == pytest.approx(k_piece(i, mz), rel=1e-14), mz
    eps = 1e-9
    for join, step in steps.items():
        jump = bottom.apl_grain_size_relations(join)["k_db_per_m_khz"] - bottom.apl_grain_size_relations(join - eps)["k_db_per_m_khz"]
        assert jump == pytest.approx(step, abs=1e-7), join
    # The last piece reaches 0.0601 at 9.5 phi to 2.5e-5: (11.3)'s constant tail (outside our -1..9 range).
    assert k_piece(4, 9.5) == pytest.approx(0.0601, abs=3e-5)


def test_fig_11_2_normal_incidence_loss():
    """Fig. 11.2's "TR 9407" curve: [N1 3.2] computes 6.1, 6.9, 10.6, 14.8, 18.5, 22.2 dB at 0..5 phi and reads
    the figure at about 6, 7, 10.5, 15, 18.5, 22 (by eye). Held at the plan's 6.1 +- 0.5, 14.8 +- 0.7,
    22.2 +- 0.7 dB at 0, 3, 5 phi, and to 0.05 dB of the note's computation."""
    got = [normal_incidence_loss_db(bottom.apl_grain_size_relations(m)) for m in range(6)]
    assert np.allclose(got, [6.1, 6.9, 10.6, 14.8, 18.5, 22.2], atol=0.05)
    for m, want, tol in ((0, 6.1, 0.5), (3, 14.8, 0.7), (5, 22.2, 0.7)):
        assert got[m] == pytest.approx(want, abs=tol)


def test_difference_from_todays_interpolation():
    """What the method changes between the tabulated points [N1 3.2], on a 0.001 phi grid: at most 0.68 dB in
    the normal-incidence reflection coefficient (at 5.3 phi), nu 0.0041 there, rho 0.0106 (at 1.25 phi),
    delta 7.6 % (near 5.79 phi); outside 5-6 phi under 0.1 dB (0.095 dB at 3.27 phi)."""
    grid = np.round(np.linspace(-1.0, 9.0, 10001), 6)
    worst = {}
    for phi in grid:
        phi = float(phi)
        new, old = bottom.sediment_from_grain_size(phi, method="relations"), bottom.sediment_from_grain_size(phi)
        diffs = {"dB": abs(normal_incidence_loss_db(new) - normal_incidence_loss_db(old)), "rho": abs(new["rho"] - old["rho"]),
                 "nu": abs(new["nu"] - old["nu"]), "delta": abs(old["delta"] / new["delta"] - 1.0)}
        if not 5.0 < phi < 6.0:
            assert diffs["dB"] < 0.1, phi
        for key, value in diffs.items():
            if value > worst.get(key, (0.0, None))[0]:
                worst[key] = (value, phi)
    assert worst["dB"][0] == pytest.approx(0.68, abs=0.005) and worst["dB"][1] == pytest.approx(5.3, abs=0.002)
    assert worst["nu"][0] == pytest.approx(0.0041, abs=5e-5) and worst["nu"][1] == pytest.approx(5.3, abs=0.002)
    assert worst["rho"][0] == pytest.approx(0.0106, abs=5e-5) and worst["rho"][1] == pytest.approx(1.25, abs=0.002)
    assert worst["delta"][0] == pytest.approx(0.076, abs=0.001) and 5.7 < worst["delta"][1] < 5.9
