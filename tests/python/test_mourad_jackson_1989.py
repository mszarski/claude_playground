"""Mourad and Jackson (1989), "High Frequency Sonar Equation Models for Bottom Backscatter and Forward
Loss", OCEANS '89 Proceedings, pp. 1168-1175 [MJ89], as extracted in
docs/references/bottom_mourad_jackson_1989.md (the paper read on the page images; status words there:
confirmed = printed in [MJ89]; figure = read off a printed curve, with its reading error; ours = computed).

C3: our volume term against the paper's eq. (39) with the misprint corrected.
"""
import math

import numpy as np
import pytest

from gpusonar.sim import bottom

C = 1500.0


def mj89_volume_39(theta, rho, nu, delta, sigma2, printed=False):
    """[MJ89] (39), typed again here from the note (Sec. 2.4), independent of bottom.py:
        sigma_pv = 5 delta sigma2 |1 - R^2(theta)|^2 sin^2 theta / (nu ln(10) |P|^2 Im P),
    P = sqrt(kappa^2 - cos^2 theta) (11), kappa = (1 + i delta) / nu (12), R = (y - 1) / (y + 1),
    y = rho sin theta / P (9)-(10). `printed=True` uses the factor |1 - R(theta)|^2 the page prints."""
    kappa = (1.0 + 1j * delta) / nu
    p = np.sqrt(kappa**2 - np.cos(theta) ** 2 + 0j)
    p = np.where(p.imag < 0.0, -p, p)
    y = rho * np.sin(theta) / p
    r = (y - 1.0) / (y + 1.0)
    factor = np.abs(1.0 - r) ** 2 if printed else np.abs(1.0 - r**2) ** 2
    return 5.0 * delta * sigma2 * factor * np.sin(theta) ** 2 / (nu * math.log(10.0) * np.abs(p) ** 2 * p.imag)


def test_volume_term_is_the_corrected_eq_39_and_not_its_print():
    """[MJ89] (39) PRINTS |1 - R(theta)|^2. That is a misprint for |1 - R^2(theta)|^2 (note Sec. 0 item 1):
    - the paper's own figures: with the printed factor its model lies 1.5-3 dB below its own solid curves
      at 20-60 deg (Figs. 6, 9, 10); with the squared R it reproduces all seven printed model curves read
      (Figs. 6-10) to rms 0.3-0.8 dB (note Sec. 3.2, figure readings +-1.5 dB; also
      the C4 figure tests in this file hold that);
    - Jackson and Richardson (2007) (14.8) and our own derivation (bottom.py module docstring) have the
      squared form: |1 - R^2|^2 sin^2 / |P|^2 = |1 + R|^4 / rho^2 (algebra, ours).
    So our `volume_strength` stays as it is: do NOT "fix" it to the printed form. Unaveraged, ours equals
    the corrected (39) to 1e-12 relative; the printed form is ours times |1 + R|^-2 exactly."""
    theta = np.radians(np.arange(5.0, 86.0, 5.0))
    for name in ("medium_sand", "very_fine_silt"):
        par = bottom.sediment(name)
        for f in (20e3, 100e3):
            ours = bottom.volume_strength(theta, theta, f, C, **par)
            corrected = mj89_volume_39(theta, par["rho"], par["nu"], par["delta"], par["sigma2"])
            assert np.allclose(ours, corrected, rtol=1e-12, atol=0.0), (name, f)
            printed = mj89_volume_39(theta, par["rho"], par["nu"], par["delta"], par["sigma2"], printed=True)
            r = bottom.reflection(theta, par["rho"], par["nu"], par["delta"])
            assert np.allclose(printed, ours / np.abs(1.0 + r) ** 2, rtol=1e-12, atol=0.0), (name, f)
            above = theta > np.radians(45.0)                                 # above the critical angle |1 + R| > 1 here,
            assert np.all(printed[above] < ours[above])                      # so the printed form is lower (by 2-5 dB)
