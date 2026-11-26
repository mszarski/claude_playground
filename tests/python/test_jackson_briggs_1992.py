"""Jackson and Briggs (1992), "High-frequency bottom backscattering: Roughness versus sediment volume scattering",
J. Acoust. Soc. Am. 92 (2), 962-977 [JB], as extracted in docs/references/bottom_backscatter_jackson_briggs_1992.md
(the paper read on the page images; status words there: confirmed = printed in [JB]; ours = computed; one site).

[JB] 5.3: a warning when the relief spectral exponent gamma2 exceeds 3.7 in the models whose result rests on the
spectrum's extrapolation to low wavenumbers (small_slope, composite, combined, williams_jackson_1998), with the
paper's reason (p. 974).
[JB] 5.5: the 1-D to 2-D spectral strength, eq. (4), against Table III -> Table V.
"""
import math
import warnings

import numpy as np
import pytest
from scipy import integrate

from gpusonar.sim import bottom

C = 1500.0
D = np.radians

# [JB] Table V (p. 973), Quinault, as printed (confirmed): along strike gamma 3.92, w2 0.00598 cm^4; across strike
# 3.67, 0.00422 cm^4; rho 1.94, nu 1.113, delta 0.0115.
QUINAULT = dict(rho=1.94, nu=1.113, delta=0.0115)
ALONG = dict(QUINAULT, w2=bottom.w2_from_cm4(0.00598, 3.92), gamma2=3.92)
ACROSS = dict(QUINAULT, w2=bottom.w2_from_cm4(0.00422, 3.67), gamma2=3.67)
WARNED = ("small_slope", "composite", "combined", "williams_jackson_1998")


@pytest.fixture(autouse=True)
def _fresh_warnings():
    bottom._warned_large_gamma2.clear()
    yield
    bottom._warned_large_gamma2.clear()


def test_gamma2_above_3p7_warns_once_per_model_with_the_papers_reason():
    """[JB] p. 974: an exponent near 4 extrapolated below the measured wavenumbers gives an unrealistic large-scale
    slope (Quinault along strike: 33 deg at 35 kHz) and too strong scattering; past applications had exponents
    "generally less than 3.7". Each of the four models warns once per process for the along-strike inputs (3.92),
    not for across strike (3.67), and the default path (no model named) is small_slope's warning. Values unchanged."""
    g = D([20.0, 30.0, 40.0])
    for model in WARNED:
        with pytest.warns(UserWarning, match=r"gamma2 = 3\.92 is above 3\.7.*Jackson and Briggs \(1992\).*33 deg"):
            first = bottom.roughness_strength(g, g, 35e3, C, roughness_model=model, **ALONG)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            again = bottom.roughness_strength(g, g, 35e3, C, roughness_model=model, **ALONG)     # once per model
            bottom.roughness_strength(g, g, 35e3, C, roughness_model=model, **ACROSS)          # 3.67: silent
        assert np.array_equal(first, again)
    bottom._warned_large_gamma2.clear()
    with pytest.warns(UserWarning, match=r"\(small_slope\)"):
        default = bottom.roughness_strength(g, g, 35e3, C, **ALONG)
    assert np.array_equal(default, bottom.small_slope_strength(g, g, 35e3, C, **ALONG))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # perturbation and kirchhoff do not extrapolate a slope: silent
        for model in ("perturbation", "kirchhoff"):
            bottom.roughness_strength(g, g, 35e3, C, roughness_model=model, **ALONG)
        for model in WARNED:                            # from 4 on: refused (no structure function), not warned
            with pytest.raises(ValueError, match="2 < gamma2 < 4"):
                bottom.roughness_strength(g, g, 35e3, C, roughness_model=model, **dict(ALONG, gamma2=4.0))


def test_the_quinault_along_strike_artefact_the_warning_is_for():
    """[JB] Sec. 4 item 12 (ours, `jb.py`): with the along-strike inputs (gamma2 3.92) our small-slope roughness at
    35 kHz is 5.6 dB above the across-strike one at 30 deg and 10.3 dB at 40 deg, the anisotropy the paper finds
    its own model predicts and its data do not show (mean levels isotropic within +-1 dB at Quinault, p. 974).
    0.1 dB (the note prints 0.1 dB)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for theta, want in ((30.0, 5.6), (40.0, 10.3)):
            g = D(theta)
            gap = 10 * math.log10(float(bottom.roughness_strength(g, g, 35e3, C, roughness_model="small_slope", **ALONG))
                                  / float(bottom.roughness_strength(g, g, 35e3, C, roughness_model="small_slope", **ACROSS)))
            assert gap == pytest.approx(want, abs=0.1), (theta, gap)
        # The paper's number for the extrapolated slope, "33 deg" at 35 kHz (confirmed, p. 974), from (A8); our split at
        # k h_s = 2^(-1/2) is (A8) (the note Sec. 0 item 2). With Table V's w2 (0.004137 m^0.08) it is 33.06 deg (ours);
        # the note's 32.9 used J&R Table 13.1's rounded 0.0041.
        slope, _ = bottom.large_scale_slope(35e3, C, w2=ALONG["w2"], gamma2=3.92, small_scale_kh=2 ** -0.5)
        assert math.degrees(math.atan(math.sqrt(2.0) * slope)) == pytest.approx(33.0, abs=0.5)


# --- [JB] 5.5 (plan G-18 in part): the 1-D to 2-D spectral strength, eq. (4) ------------------------------------

# [JB] Table III (p. 965): 1-D regression slope and intercept at 1 cycle/cm (cm^3); Table V (p. 973): the 2-D gamma
# and w2 (cm^4 at 1 rad/cm) the paper used. All as printed (confirmed).
TABLE_III_V = {
    "Quinault along strike": (-2.92, 0.28e-3, 3.92, 0.00598),
    "Quinault across strike": (-2.67, 0.33e-3, 3.67, 0.00422),
    "Arafura Sea": (-2.18, 0.69e-3, 3.18, 0.00318),
    "San Francisco along, pre-storm": (-2.65, 0.72e-4, 3.65, 0.000846),
    "San Francisco along, post-storm": (-2.73, 0.57e-4, 3.73, 0.000826),
    "San Francisco across, pre-storm": (-2.38, 1.30e-4, 3.38, 0.000912),
    "San Francisco across, post-storm": (-2.56, 1.27e-4, 3.56, 0.001296),
}
PROBABLE_MISPRINT = "San Francisco along, pre-storm"


def test_eq_4_gives_table_v_from_table_iii_for_six_rows_and_records_the_seventh():
    """(4): w2 = (2 pi)^(gamma - 2) phi_1 h0 Gamma(gamma / 2) / [pi^(1/2) Gamma((gamma - 1) / 2)], gamma = 1 - slope.
    Six rows agree with Table V within 0.11 % (bound 0.2 %). The seventh, San Francisco pre-storm along strike, gives
    0.000885 against the printed 0.000846 (+4.6 %, 0.19 dB): RECORDED, not "fixed" - which of the two printed numbers
    is wrong is not known (the note Sec. 6); the site row keeps the printed Table V value."""
    for name, (slope, intercept, gamma_v, w2_v) in TABLE_III_V.items():
        got = bottom.roughness_2d_from_1d(intercept, slope)
        assert got["gamma2"] == pytest.approx(gamma_v, abs=1e-12), name
        if name == PROBABLE_MISPRINT:
            assert got["w2_cm4"] == pytest.approx(0.000885, abs=5e-7) and got["w2_cm4"] / w2_v == pytest.approx(1.0456, abs=0.001)
        else:
            assert got["w2_cm4"] == pytest.approx(w2_v, rel=2e-3), (name, got["w2_cm4"] / w2_v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert bottom.sediment_table()["jackson_briggs_1992_san_francisco"]["scattering"]["w2_cm4"] == 0.000846


def test_eq_4_is_the_integral_of_the_2d_spectrum_over_the_cross_wavenumber():
    """What (4) stands for (ours, as the note's `jb.py`): the two-sided 1-D spectrum per cycle/cm along a line,
    Phi_1(f) = 2 pi integral W(sqrt(Kx^2 + Ky^2)) dKy at Kx = 2 pi f, of the 2-D law W(K) = w2 K^-gamma (K in rad/cm),
    equals intercept x f^slope. By quadrature, three exponents, 1e-8 relative."""
    for slope, intercept in ((-2.18, 0.69e-3), (-2.65, 0.72e-4), (-2.92, 0.28e-3)):
        rel = bottom.roughness_2d_from_1d(intercept, slope)
        gamma, w2 = rel["gamma2"], rel["w2_cm4"]
        for f in (0.1, 1.0, 3.0):
            kx = 2 * math.pi * f
            inner = integrate.quad(lambda ky: w2 * (kx * kx + ky * ky) ** (-gamma / 2), 0.0, np.inf, epsabs=0.0, epsrel=1e-12)[0]
            assert 2 * math.pi * 2 * inner == pytest.approx(intercept * f ** slope, rel=1e-8), (slope, f)


def test_eq_4_refuses_a_spectrum_that_does_not_fall_fast_enough():
    """The 1-D integral over Ky exists only for gamma > 1 (slope < 0); a 2-D model needs 2 < gamma < 4 besides, which
    the models check themselves."""
    for slope in (0.0, 0.5):
        with pytest.raises(ValueError, match="slope"):
            bottom.roughness_2d_from_1d(1e-4, slope)
