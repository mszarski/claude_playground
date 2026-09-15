"""The normal-mode reference, and hydropt's ray model checked against it.

Two kinds of test here.  The first checks the *reference* is right -- eigenvalues
satisfying the characteristic equation, modes orthonormal and matching their
boundary conditions, and the field's prefactor verified against the method of
images rather than recalled.  Only then is it worth anything as a check on
something else.

The second is the cross-check itself: hydropt's incoherent ray sum against the
exact image-source eigenray energies in an ideal waveguide.  That is the first
test in the suite where ray theory is compared against a formulation that is not
ray theory, and it found a real defect in how absolute levels are rendered -- see
`test_the_default_spreading_is_applied_twice_in_a_dense_fan` and the README.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import hankel1, kv

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, make_time_grid, splat_etc,
    structured_fan, trace,
)
from hydropt.pekeris import (
    PekerisWaveguide, hankel1_outgoing, ideal_image_field, ideal_mode_field,
)

H, C1, C2, RHO1, RHO2 = 100.0, 1500.0, 1800.0, 1000.0, 1800.0


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _wg() -> PekerisWaveguide:
    return PekerisWaveguide(H, C1, C2, RHO1, RHO2)


# --------------------------------------------------------------------------- #
# Is the reference itself right?
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq", [50.0, 100.0, 200.0, 500.0])
def test_eigenvalues_satisfy_the_characteristic_equation(freq):
    wg = _wg()
    gamma = wg.gammas(freq)
    assert gamma.size == wg.n_modes(freq) > 0
    residual = np.abs(wg._characteristic(gamma, freq))
    # Scale: the function is O(rho * gamma) ~ O(1e3), so 1e-9 absolute is a
    # relative residual near machine precision.
    assert residual.max() < 1e-8, f"worst residual {residual.max():.2e}"


@pytest.mark.parametrize("freq", [50.0, 200.0, 500.0])
def test_every_trapped_mode_lies_between_the_two_wavenumbers(freq):
    """``k2 < k_r < k1`` is what *trapped* means; outside it the mode is leaky."""
    wg = _wg()
    k1, k2 = 2 * math.pi * freq / C1, 2 * math.pi * freq / C2
    k_r = wg.wavenumbers(freq)
    assert np.all(k_r > k2) and np.all(k_r < k1)
    assert np.all(np.diff(k_r) < 0), "wavenumbers should come out descending"


def test_the_mode_count_is_the_ray_critical_angle_formula():
    """Where the two pictures meet: ``M = floor(2 H sin(theta_c) / lambda + 1/2)``.

    A ray steeper than ``theta_c`` refracts into the bottom and is lost; a mode
    with ``k_r < k2`` is leaky for the same reason, so the count of trapped modes
    is fixed by the ray critical angle and the depth in wavelengths.
    """
    wg = _wg()
    sin_c = math.sin(math.radians(wg.critical_angle_deg()))
    for freq in (30.0, 75.0, 150.0, 300.0, 600.0):
        lam = C1 / freq
        predicted = int(math.floor(2 * H * sin_c / lam + 0.5))
        assert wg.n_modes(freq) == predicted, f"{freq} Hz"
        assert len(wg.gammas(freq)) == predicted


def test_mode_one_cutoff_is_where_the_first_mode_appears():
    wg = _wg()
    f_c = wg.cutoff_frequency_hz(1)
    assert wg.n_modes(f_c * 0.9) == 0
    assert wg.n_modes(f_c * 1.1) >= 1
    for m in (2, 5, 9):
        f_m = wg.cutoff_frequency_hz(m)
        assert wg.n_modes(f_m * 0.999) == m - 1
        assert wg.n_modes(f_m * 1.001) == m


def test_group_velocity_sits_below_the_water_speed_and_obeys_vg_vp_of_c_squared():
    """A guide is dispersive, and the group velocity is *below* ``c_water``.

    My first version of this test asserted ``vg > c_water``, which is wrong
    physics: for an ideal waveguide ``vg vp = c^2`` exactly, and since ``vp > c``
    the group velocity must be below it.  Energy travels slower than the sound
    speed because it zig-zags.  The Pekeris guide departs from ``vg vp = c^2``
    only through the penetrable bottom, so the product is close but not exact.
    """
    wg = _wg()
    for freq in (100.0, 300.0):
        vp, vg = wg.phase_velocity(freq), wg.group_velocity(freq)
        assert np.all(vp > C1) and np.all(vp < C2 * 1.001)
        assert np.all(vg > 0.0) and np.all(vg < C1), f"{freq} Hz: {vg}"
        assert np.all(vg < vp[: len(vg)]), "normal dispersion"
        product = vp[: len(vg)] * vg / (C1 * C1)
        assert np.all(product > 0.75) and np.all(product < 1.02), product

    ideal_like = PekerisWaveguide(H, C1, 100_000.0, RHO1, 1e9)  # nearly rigid
    vp, vg = ideal_like.phase_velocity(300.0), ideal_like.group_velocity(300.0)
    n = len(vg)
    assert np.abs(vp[:n] * vg / (C1 * C1) - 1.0).max() < 0.02


def test_modes_are_orthonormal_under_the_density_weighted_inner_product():
    """``int Z_m Z_n / rho dz = delta_mn``, integrated numerically over water *and*
    the evanescent tail -- which carries real energy, so omitting it from the
    normalisation would misstate every level."""
    wg = _wg()
    freq = 200.0
    z = np.linspace(0.0, H + 400.0, 400_001)
    rho = np.where(z <= H, RHO1, RHO2)
    shapes = wg.mode_shapes(freq, z)
    gram = np.trapezoid(shapes[:, None, :] * shapes[None, :, :] / rho, z, axis=-1)
    assert np.abs(np.diag(gram) - 1.0).max() < 2e-5
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() < 2e-5


def test_modes_vanish_at_the_pressure_release_surface():
    shapes = _wg().mode_shapes(200.0, np.array([0.0]))
    assert np.abs(shapes[:, 0]).max() < 1e-15


def test_pressure_and_weighted_gradient_are_continuous_at_the_seabed():
    """The two conditions the characteristic equation came from.

    Tested by convergence rather than by a fixed tolerance: evaluating either
    side at ``H +/- eps`` differs from the interface value by ``O(eps)`` whatever
    the boundary condition, so the meaningful check is that the jump *shrinks
    linearly* as ``eps`` does.  A genuine discontinuity would not.
    """
    wg = _wg()
    freq = 200.0
    jumps, grad_jumps = [], []
    for eps in (1e-3, 1e-4, 1e-5):
        above = wg.mode_shapes(freq, np.array([H - eps]))[:, 0]
        below = wg.mode_shapes(freq, np.array([H + eps]))[:, 0]
        jumps.append(np.abs(above - below).max())
        d_above = (wg.mode_shapes(freq, np.array([H - eps]))[:, 0]
                   - wg.mode_shapes(freq, np.array([H - 3 * eps]))[:, 0]) / (2 * eps)
        d_below = (wg.mode_shapes(freq, np.array([H + 3 * eps]))[:, 0]
                   - wg.mode_shapes(freq, np.array([H + eps]))[:, 0]) / (2 * eps)
        grad_jumps.append(np.abs(d_above / RHO1 - d_below / RHO2).max())
    for a, b in zip(jumps, jumps[1:]):
        assert b < a / 5.0, f"pressure jump not shrinking with eps: {jumps}"
    assert grad_jumps[-1] < 1e-6, f"weighted-gradient jump {grad_jumps}"


def test_a_slower_bottom_traps_nothing_and_says_so():
    slow = PekerisWaveguide(H, 1500.0, 1480.0)
    with pytest.raises(ValueError, match="not the faster medium"):
        slow.critical_angle_deg()
    assert slow.n_modes(200.0) == 0
    assert slow.wavenumbers(200.0).size == 0
    assert np.all(np.isinf(slow.transmission_loss(200.0, np.array([1000.0]), 50.0, 50.0)))


# --------------------------------------------------------------------------- #
# The prefactor, verified rather than recalled
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq,zs,zr", [(75.0, 10.0, 90.0), (175.0, 30.0, 70.0),
                                        (300.0, 50.0, 50.0)])
def test_the_mode_prefactor_matches_the_method_of_images(freq, zs, zr):
    """Two expansions of the same Green's function must agree.

    This is what pins ``i pi / rho`` in :meth:`PekerisWaveguide.pressure`: a stray
    ``4 pi`` or ``sqrt(2)`` would show up here immediately, and nothing else in
    the suite would notice it.
    """
    ranges = np.array([300.0, 800.0, 2000.0])
    modes = ideal_mode_field(H, C1, freq, ranges, zs, zr, n_modes=3000)
    images = ideal_image_field(H, C1, freq, ranges, zs, zr, n_images=20_000)
    rel = np.abs(modes - images) / np.abs(images)
    assert rel.max() < 1e-7, f"worst relative disagreement {rel.max():.2e}"


def test_a_mode_exactly_at_cutoff_is_dropped_rather_than_infinite():
    """H = 100 m, c = 1500 m/s, f = 150 Hz puts mode 20 exactly at cutoff, where
    ``k_r = 0`` and the expansion is logarithmically singular.  This is how the
    case was found: the first version returned NaN for the whole field."""
    field = ideal_mode_field(H, C1, 150.0, np.array([500.0, 1500.0]), 30.0, 70.0)
    assert np.all(np.isfinite(field))
    images = ideal_image_field(H, C1, 150.0, np.array([500.0, 1500.0]), 30.0, 70.0,
                              n_images=20_000)
    assert np.abs(field - images).max() / np.abs(images).max() < 1e-7


def test_hankel1_outgoing_handles_imaginary_arguments_without_overflow():
    """``J_0 + i Y_0`` each grow like ``e^x`` for imaginary argument while the
    combination decays, so the naive evaluation loses the answer to cancellation
    long before it overflows.  The ``K_0`` identity gives it directly."""
    real = np.array([0.5, 2.0, 50.0], dtype=complex)
    assert np.allclose(hankel1_outgoing(real), hankel1(0, real.real), rtol=1e-12)
    imag = 1j * np.array([1.0, 10.0, 400.0])
    got = hankel1_outgoing(imag)
    assert np.all(np.isfinite(got))
    assert np.allclose(got, -2j / math.pi * kv(0, imag.imag), rtol=1e-12)
    assert abs(got[-1]) < 1e-100, "a deeply evanescent mode should vanish, not blow up"


# --------------------------------------------------------------------------- #
# The cross-check: hydropt's rays against exact eigenray energies
# --------------------------------------------------------------------------- #
def _ideal_scenes(n_steps: int):
    common = dict(field=IsoProfile(C1, learnable=False),
                  source=(0.0, 0.0, 50.0),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([0.2]), step_size=10.0, n_steps=n_steps)
    free = Scene(surface=FlatHeight(-1e5), bottom=FlatHeight(1e5), max_bounces=0,
                 **common)
    guide = Scene(surface=FlatHeight(0.0), bottom=FlatHeight(H), max_bounces=200,
                  **common)
    return free, guide


def _render(scene, fan, weights, rx, grid, spreading="unit"):
    result = trace(scene, fan)
    sp = torch.ones_like(result.arclen) if spreading == "unit" else None
    return splat_etc(result, rx, grid, torch.tensor([0.2]), sigma_d=5.0,
                     sigma_t=2e-4, ray_weights=weights, spreading=sp,
                     absorption=lambda f: torch.zeros_like(f),
                     ray_chunk=2000)[:, 0].sum(-1)


def _fan(n_e=440, n_a=20, elev=30.0, azim=3.5):
    dirs, e, a = structured_fan(n_e, n_a, elev_range_deg=(-elev, elev),
                                azim_range_deg=(-azim, azim))
    de = math.radians(2 * elev) / (n_e - 1)
    da = math.radians(2 * azim) / (n_a - 1)
    w = (torch.cos(e).reshape(-1, 1) * de * da).expand(n_e, n_a).reshape(-1)
    return dirs, w.contiguous(), elev


def test_ray_eigenray_energies_match_the_exact_image_source_solution():
    """The headline cross-check, and the first one not made against ray theory.

    In an ideal lossless waveguide every eigenray is an image, with energy
    ``1/R^2`` exactly.  Calibrating hydropt's arbitrary scale *once* in free
    space and transferring it unchanged, each resolved eigenray matches to five
    or six figures across 0 to 30 degrees grazing.
    """
    r = 700.0
    n_steps = 130
    dirs, w, elev_max = _fan()
    free, guide = _ideal_scenes(n_steps)
    grid = make_time_grid(0.46, 0.63, 7000)
    assert float(grid[1] - grid[0]) < 2e-4 / 4, "time bins must resolve sigma_t"

    k = float(_render(free, dirs, w, torch.tensor([[r, 0.0, 50.0]]), grid)[0]) * r * r
    # One receiver, so integrate the fine-grid ETC per arrival.
    result = trace(guide, dirs)
    etc = splat_etc(result, torch.tensor([[r, 0.0, 50.0]]), grid,
                    torch.tensor([0.2]), sigma_d=5.0, sigma_t=2e-4, ray_weights=w,
                    spreading=torch.ones_like(result.arclen),
                    absorption=lambda f: torch.zeros_like(f), ray_chunk=2000)[0, 0]

    checked = 0
    for m in range(0, 6):
        dz = 100.0 * m
        path = math.hypot(r, dz)
        if math.degrees(math.atan2(dz, r)) > elev_max - 2.0:
            break
        exact = (1 if m == 0 else 2) / (path * path)
        t = path / C1
        window = (grid >= t - 8 * 2e-4) & (grid <= t + 8 * 2e-4)
        got = float(etc[window].sum()) / k
        assert got / exact == pytest.approx(1.0, rel=2e-3), (
            f"eigenray m={m} at {math.degrees(math.atan2(dz, r)):.1f} deg: "
            f"ray {got:.6e} vs exact {exact:.6e}")
        checked += 1
    assert checked >= 4, f"only {checked} eigenrays resolved"


def test_the_default_spreading_is_applied_twice_in_a_dense_fan():
    """A real defect, found by the cross-check and pinned here.

    Each ray carries ``1/s^2``, *and* the number of rays landing inside the fixed
    ``sigma_d`` acceptance also falls as ``1/s^2`` -- so a dense-fan sum applies
    geometric spreading twice and ETC energy falls as ``1/R^4``.  Passing unit
    ``spreading`` lets the ray *count* supply the spreading, which is exact (see
    the test above).  The README documents what this does and does not affect.
    """
    # Ranges where the fan is wide enough to contain the 6-sigma acceptance gate:
    # at 400 m a +/-3 deg fan spans only +/-21 m against a 30 m gate, and the
    # truncation shows up as an exponent near 3.7 that is neither 2 nor 4.
    ranges = [800.0, 1600.0, 3200.0]
    dirs, w, _ = _fan(n_e=150, n_a=150, elev=3.0, azim=3.0)
    free, _ = _ideal_scenes(600)
    rx = torch.stack([torch.tensor(ranges), torch.zeros(3),
                      torch.full((3,), 50.0)], dim=-1)
    grid = make_time_grid(0.4, 2.4, 9000)

    def exponent(spreading):
        e = _render(free, dirs, w, rx, grid, spreading=spreading)
        return [math.log(float(e[i] / e[i + 1])) / math.log(ranges[i + 1] / ranges[i])
                for i in range(2)]

    default = exponent("default")
    counted = exponent("unit")
    assert all(abs(x - 4.0) < 0.05 for x in default), f"default gave {default}"
    assert all(abs(x - 2.0) < 0.05 for x in counted), f"unit gave {counted}"
