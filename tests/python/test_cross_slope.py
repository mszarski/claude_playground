"""The cross-slope warning of an N x 2-D run (docs/tasks/PROPAGATION_3D.md, option B, package P1).

Our eigenray tier traces one depth profile per bearing. Where the bed slopes ACROSS a bearing a reflected ray leaves
the bearing's vertical plane by phi = atan(tan g sin 2 alpha) (ours, the document's section 3), which N x 2-D cannot
follow; over the 12 degree flanks of the seamount of test_seamount_3d.py that moved the level by 3 to 4 dB (D-149,
D-197). The reverberation over relief (reverb_relief.py, the only place a 3-D bed reaches the eigenray tier) now says
so, and records the largest turn on the provider as `cross_slope`. It changes no number: a warning only.
"""

import warnings

import numpy as np
import pytest

from gpusonar.sim import FreeFieldProvider, bathymetry, reverb_relief

SEAMOUNT = {"seamount": dict(depth=200.0, height=180.0, center=[2500.0, 300.0], sigma=500.0)}


def test_turn_formula_against_the_documents_table():
    """The table of section 3 (degrees): alpha 1 deg at g 5 deg 0.17; 12 deg at 5 deg 2.0, at 45 deg 22.1."""
    turn = lambda alpha, g: np.degrees(bathymetry.out_of_plane_turn(np.tan(np.radians(alpha)), np.radians(g)))   # noqa: E731
    assert turn(1.0, 5.0) == pytest.approx(0.17, abs=0.005)
    assert turn(12.0, 5.0) == pytest.approx(2.0, abs=0.05)
    assert turn(12.0, 45.0) == pytest.approx(22.1, abs=0.05)
    assert turn(-12.0, 5.0) == pytest.approx(-turn(12.0, 5.0))
    assert turn(0.0, 30.0) == 0.0


def test_cross_bearing_slope_of_a_plane():
    bed = bathymetry.plane(50.0, (0.1, 0.0))                      # deepens northwards
    r = np.array([0.0, 100.0])
    assert np.allclose(bathymetry.cross_bearing_slope(bed, (0.0, 0.0), 0.0, r), 0.0)                 # along the slope
    assert np.allclose(bathymetry.cross_bearing_slope(bed, (0.0, 0.0), np.pi / 2, r), -0.1)          # bearing east: right is south
    assert np.allclose(bathymetry.cross_bearing_slope(bed, (0.0, 0.0), -np.pi / 2, r), 0.1)


def test_seamount_flanks_are_flagged_and_beside_is_not():
    """Bearings and range of test_seamount_3d.py (source at the origin, 6.5 km)."""
    bearings = np.array([-20.0, -4.0, 6.0, 16.0, 32.0])
    rep = reverb_relief.cross_slope_report(bathymetry.make_bathymetry(SEAMOUNT), (0.0, 0.0), np.radians(bearings), 6500.0)
    flagged = dict(zip(bearings, rep["flagged"]))
    assert flagged[-4.0] and flagged[16.0]                         # the flanks: 3.0 to 3.6 dB of 3-D effect
    assert not flagged[-20.0] and not flagged[32.0]                # beside
    assert not flagged[6.0]                                        # straight at the summit: the slope is along it
    assert rep["slope_deg"].max() == pytest.approx(12.3, abs=0.1) and rep["turn_deg"].max() > 4.0


def test_level_bed_and_a_range_profile_about_the_sonar_are_never_flagged():
    az = np.radians(np.arange(0.0, 360.0, 5.0))
    level = bathymetry.make_bathymetry({"plane": {"depth": 50.0}})
    assert not reverb_relief.cross_slope_report(level, (0.0, 0.0), az, 3000.0)["flagged"].any()
    # The fjord scenarios' bed (configs/scenarios/fjord_*: one range profile on every bearing, about the sonar).
    fjord = bathymetry.make_bathymetry({"range": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                                        "depth": [50, 57.1, 72.3, 84.3, 84.3, 72.8, 58.9, 53, 58.3, 68.4, 76]})
    rep = reverb_relief.cross_slope_report(fjord, (0.0, 0.0), az, 1000.0)
    assert not rep["flagged"].any() and rep["slope_deg"].max() < 0.1


class _Fan(FreeFieldProvider):
    """A provider that the reverberation over relief runs N x 2-D (it has `along_profile`), with free-field rays."""

    def along_profile(self, profile):
        return self


def _run(provider, bed):
    rx = np.array([0.0, 0.0, 5.0])
    return reverb_relief.directional_power(provider, rx, rx, bed, np.arange(0.02, 0.1, 2e-3),
                                           np.radians([0.0, 90.0, 180.0, 270.0]), frequency=8e3, mu_db=-27.0)


def test_the_n_by_2d_reverberation_warns_and_records_and_changes_nothing():
    steep = bathymetry.plane(30.0, (0.0, 0.2))                     # deepens eastwards by 11.3 deg: across north and south
    fan = _Fan(1500.0)
    with pytest.warns(RuntimeWarning, match="slopes across the bearing"):
        warned = _run(fan, steep)
    assert fan.cross_slope["slope_deg"] == pytest.approx(11.31, abs=0.01) and fan.cross_slope["bearings_flagged"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = _run(FreeFieldProvider(1500.0), steep)            # not N x 2-D: straight rays to the facets, no warning
        _run(_Fan(1500.0), bathymetry.plane(30.0, (0.01, 0.0)))   # a gentle slope: no warning
    assert np.array_equal(warned["power"], plain["power"])        # the warning changes no number
