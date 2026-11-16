"""PICMUS plane-wave benchmark (simulation, resolution phantom) through our kernel.

Needs the data: `tools/fetch_picmus.py <dir>` then set GPUSONAR_PICMUS_DIR=<dir>.
Skipped when the data is not there.
"""

import os
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.benchmarks import picmus_plane_wave_subset, picmus_resolution
from gpusonar.sources import PicmusSource

import reference_bp as ref

_DIR = os.environ.get("GPUSONAR_PICMUS_DIR", "")
_FILE = pathlib.Path(_DIR) / "resolution_distorsion_simu_dataset_iq.hdf5"
pytestmark = pytest.mark.skipif(not (_DIR and _FILE.exists()),
                                reason="PICMUS data not available (set GPUSONAR_PICMUS_DIR)")


def test_records_are_valid_plane_wave_pings():
    src = PicmusSource(str(_FILE))
    assert len(src) == 75
    rec = src.read(0)
    assert gs.validate(rec).ok, str(gs.validate(rec))
    assert rec.transmit.kind == gs.TransmitKind.PlaneWave
    assert rec.samples.n_channels == 128
    assert np.degrees(src.angles[0]) == pytest.approx(-16.0, abs=1e-3)
    # Every transmit's samples start when its wavefront crosses the array centre.
    assert all(abs(src.read(i).samples.start_time) < 1e-7 for i in (0, 20, 37, 74))


def test_plane_wave_kernel_matches_reference_on_real_data():
    src = PicmusSource(str(_FILE))
    records = [src.read(i) for i in (3, 37, 70)]
    pix = np.array([[0.0202, 0.0, 0.0], [0.0202, 0.005, 0.0], [0.0152, 0.0002, 0.0], [0.03, -0.01, 0.0]])
    got = gs.core.backproject(records, pix, interp="sinc")
    want = ref.backproject(records, pix, interp="sinc")
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-4


def test_resolution_and_coherent_compounding():
    one = picmus_resolution(str(_FILE), 1)
    full = picmus_resolution(str(_FILE), 75)
    wavelength = full["wavelength"]
    # All eleven point targets are found, on the phantom's 5 mm lattice.
    pos = np.array(full["positions"]) * 1e3
    np.testing.assert_allclose(np.diff(pos[:5, 0]), 5.0, atol=0.06)
    np.testing.assert_allclose(pos[5:, 1], [-15, -10, -5, 5, 10, 15], atol=0.08)
    # Axial width is set by the pulse: the same for 1 and 75 plane waves, about 1.3 wavelengths.
    assert full["axial_mean"] == pytest.approx(one["axial_mean"], rel=0.03)
    assert 1.1 * wavelength < full["axial_mean"] < 1.6 * wavelength
    # Compounding 75 angles coherently must not blur the points. (It did, by a factor of
    # three in the axial direction, before the per-angle time origin was handled.)
    assert full["lateral_mean"] <= 1.02 * one["lateral_mean"]
    assert not np.any(np.isnan(full["axial"])) and not np.any(np.isnan(full["lateral"]))


def test_receive_f_number_trades_resolution_for_fewer_elements():
    wide = picmus_resolution(str(_FILE), 11)
    f175 = picmus_resolution(str(_FILE), 11, f_number=1.75)
    assert f175["lateral_mean"] > 1.3 * wide["lateral_mean"]
    assert f175["axial_mean"] == pytest.approx(wide["axial_mean"], rel=0.05)


def test_subset_selection():
    assert picmus_plane_wave_subset(1) == [37]
    s = picmus_plane_wave_subset(11)
    assert len(s) == 11 and s[0] == 0 and s[-1] == 74 and 37 in s
