"""PICMUS plane-wave benchmark (simulation: resolution and contrast phantoms) through our kernel, and our port of the
challenge's own evaluation code (gpusonar.benchmarks, from the challenge archive's us_contrast.m and us_resolution.m).

Needs the data: `tools/fetch_picmus.py <dir> --reference` then set GPUSONAR_PICMUS_DIR=<dir>.
The tests that need it are skipped when it is not there; the metric tests on synthetic images always run.
"""

import os
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import benchmarks as bm
from gpusonar.benchmarks import picmus_plane_wave_subset, picmus_resolution
from gpusonar.sources import PicmusSource

import reference_bp as ref

_DIR = os.environ.get("GPUSONAR_PICMUS_DIR", "")
_FILE = pathlib.Path(_DIR) / "resolution_distorsion_simu_dataset_iq.hdf5"
_CONTRAST = pathlib.Path(_DIR) / "contrast_speckle_simu_dataset_iq.hdf5"
_REF = {k: pathlib.Path(_DIR) / f"{k}_simu_img_from_iq.hdf5" for k in ("contrast_speckle", "resolution_distorsion")}
needs_data = pytest.mark.skipif(not (_DIR and _FILE.exists() and _CONTRAST.exists()),
                                reason="PICMUS data not available (set GPUSONAR_PICMUS_DIR; tools/fetch_picmus.py)")
needs_reference = pytest.mark.skipif(not (_DIR and all(f.exists() for f in _REF.values())),
                                     reason="PICMUS reference images not available (set GPUSONAR_PICMUS_DIR; "
                                            "tools/fetch_picmus.py --reference)")


@needs_data
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


@needs_data
def test_plane_wave_kernel_matches_reference_on_real_data():
    src = PicmusSource(str(_FILE))
    records = [src.read(i) for i in (3, 37, 70)]
    pix = np.array([[0.0202, 0.0, 0.0], [0.0202, 0.005, 0.0], [0.0152, 0.0002, 0.0], [0.03, -0.01, 0.0]])
    got = gs.core.backproject(records, pix, interp="sinc")
    want = ref.backproject(records, pix, interp="sinc")
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-4


@needs_data
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


@needs_data
def test_receive_f_number_trades_resolution_for_fewer_elements():
    wide = picmus_resolution(str(_FILE), 11)
    f175 = picmus_resolution(str(_FILE), 11, f_number=1.75)
    assert f175["lateral_mean"] > 1.3 * wide["lateral_mean"]
    assert f175["axial_mean"] == pytest.approx(wide["axial_mean"], rel=0.05)


def test_subset_selection():
    assert picmus_plane_wave_subset(1) == [37]
    s = picmus_plane_wave_subset(11)
    assert len(s) == 11 and s[0] == 0 and s[-1] == 74 and 37 in s


# --- the challenge's evaluation, ported (gpusonar.benchmarks) ----------------------------------------------------

def _grid():
    x, z = bm.picmus_grid()
    xx, zz = np.meshgrid(x, z)
    return x, z, xx, zz


def test_challenge_cnr_uses_the_inside_disc_and_the_outer_ring_only():
    # Inside the cyst -20 dB, around it 0 dB, each with +-1 dB alternating; the band between the two masks and
    # everything beyond the ring at -100 dB, which must not count. |mean difference| 20 dB over a sigma of 1 dB.
    _, _, xx, zz = _grid()
    xc, zc = 0.0, 30e-3
    r, lr = bm.PICMUS_CYST_DIAMETER_SIMU / 2.0, bm.PICMUS_LATERAL_RESOLUTION_SIMU
    rin, rout1 = r - lr, r + lr
    rout2 = 1.2 * np.hypot(rin, rout1)
    d = np.hypot(xx - xc, zz - zc)
    wobble = np.where((np.arange(xx.size).reshape(xx.shape) % 2) == 0, 1.0, -1.0)
    db = np.full(xx.shape, -100.0)
    db[d <= rin] = -20.0 + wobble[d <= rin]
    ring = (d >= rout1) & (d <= rout2)
    db[ring] = wobble[ring]
    cnr = bm.picmus_cnr(10.0 ** (db / 20.0), cysts=[(xc, zc)])
    assert cnr[0] == pytest.approx(20.0 * np.log10(20.0), abs=0.11)
    with pytest.raises(ValueError, match="grid"):
        bm.picmus_cnr(np.ones((10, 10)))


def test_challenge_width_of_a_gaussian_spot():
    # A Gaussian envelope: -6 dB of the envelope is half its peak, so the width is 2 sqrt(2 ln 2) sigma.
    x, z, xx, zz = _grid()
    sx, sz = 0.25e-3, 0.17e-3
    env = 1e-6 + np.exp(-0.5 * (((xx - 5e-3) / sx) ** 2 + ((zz - 20e-3) / sz) ** 2))
    w = bm.picmus_challenge_resolution(env, points=[(5e-3, 20e-3)])
    k = 2.0 * np.sqrt(2.0 * np.log(2.0)) * 1e3
    assert w[0, 0] == pytest.approx(k * sz, abs=0.02) and w[0, 1] == pytest.approx(k * sx, abs=0.03)


def _reference(kind):
    import h5py
    with h5py.File(_REF[kind], "r") as f:
        g = f["US/US_DATASET0000"]
        env = np.abs(g["data/real"][()] + 1j * g["data/imag"][()])        # [frame, x, z]
        return [int(n) for n in g["number_plane_waves"][()]], np.transpose(env, (0, 2, 1))


@needs_reference
def test_our_port_of_the_challenge_evaluation_reproduces_its_outputs():
    """The challenge archive's evaluation/*_simu_evaluation_from_iq.txt for its own reference images, every number."""
    import h5py
    with h5py.File(pathlib.Path(_DIR) / "contrast_speckle_simu_phantom.hdf5", "r") as f:
        g = f["US/US_DATASET0000"]
        cysts = np.column_stack([g["phantom_occlusionCenterX"][()], g["phantom_occlusionCenterZ"][()]]) * 1e3
        np.testing.assert_allclose(cysts, bm.PICMUS_CYSTS_SIMU_MM, atol=1e-4)
        assert float(g["phantom_occlusionDiameter"][0]) == pytest.approx(bm.PICMUS_CYST_DIAMETER_SIMU)
        assert float(g["phantom_lateralResolution"][0]) == pytest.approx(bm.PICMUS_LATERAL_RESOLUTION_SIMU, rel=1e-6)
    with h5py.File(pathlib.Path(_DIR) / "resolution_distorsion_simu_phantom.hdf5", "r") as f:
        sca = f["US/US_DATASET0000/scatterers_positions"][()].T[:, [0, 2]] * 1e3
        np.testing.assert_allclose(sca, bm.PICMUS_SCATTERERS_SIMU_MM, atol=1e-4)
    with h5py.File(pathlib.Path(_DIR) / "contrast_speckle_simu_scan.hdf5", "r") as f:
        x, z = bm.picmus_grid()
        np.testing.assert_allclose(f["US/US_DATASET0000/x_axis"][()], x, atol=1e-12)
        np.testing.assert_allclose(f["US/US_DATASET0000/z_axis"][()], z, atol=1e-12)
    # Contrast, in the file's order (middle, left, right column; shallow to deep), for 1, 3, 11 and 75 plane waves.
    want = {1: [9.6, 10.5, 10.5, 8.8, 10.0, 9.4, 8.8, 10.2, 9.7], 3: [6.1, 9.3, 10.9, 6.8, 7.1, 8.8, 6.3, 6.9, 9.3],
            11: [10.9, 13.4, 15.1, 11.1, 11.4, 12.6, 11.0, 11.8, 13.0], 75: [16.1, 16.0, 15.6, 15.4, 15.1, 14.0, 15.7, 15.4, 14.9]}
    counts, env = _reference("contrast_speckle")
    for n, e in zip(counts, env):
        cnr = bm.picmus_cnr(e)
        np.testing.assert_allclose(cnr, want[n], atol=1e-9)
        if n in bm.PICMUS_REFERENCE_DAS_SIMU:
            assert round(float(cnr.mean()), 2) == bm.PICMUS_REFERENCE_DAS_SIMU[n]["cnr_db"]
    # Resolution: the printed means (two decimals) and the 75-wave lateral widths of the vertical column.
    counts, env = _reference("resolution_distorsion")
    for n, e in zip(counts, env):
        w = bm.picmus_challenge_resolution(e)
        if n in bm.PICMUS_REFERENCE_DAS_SIMU:
            ref = bm.PICMUS_REFERENCE_DAS_SIMU[n]
            assert round(float(w[:, 0].mean()), 2) == ref["axial_mm"] and round(float(w[:, 1].mean()), 2) == ref["lateral_mm"]
        if n == 75:
            np.testing.assert_allclose(np.round(w[:8, 1], 2), [0.53, 0.53, 0.53, 0.53, 0.53, 0.54, 0.54, 0.55])


@pytest.fixture(scope="module")
def ours_f175():
    """Our kernel at the challenge's receive F-number (1.75, hard edge) for 1, 11 and 75 plane waves."""
    return {n: (bm.picmus_contrast(str(_CONTRAST), n), bm.picmus_challenge_widths(str(_FILE), n)) for n in (1, 11, 75)}


@needs_data
def test_contrast_and_resolution_against_the_challenges_reference_das(ours_f175):
    """Against what the challenge's own evaluation printed for its reference DAS (Tukey 25 % receive window where ours
    has a hard edge at the same F-number 1.75).

    The bounds are set from what was observed when this test was written, with headroom; they are not derived:
    - CNR at most 1.5 dB below the reference: observed -1.09 dB (1 wave), -0.64 (11), -0.23 (75);
    - lateral width 0.85 to 1.0 of the reference: observed ratio 0.895 (1 wave), 0.956 (11), 0.955 (75).
    "Never above" (CNR not higher, lateral width not wider) rests on an argument, not a proof: an untapered aperture
    edge gives a narrower main lobe and higher sidelobes than the same aperture tapered, so more sidelobe clutter
    fills the cysts (lower CNR) and the points are narrower.
    The axial check is a sanity check only: every configuration, ours and the reference, gives 0.40 mm (the pulse
    sets it), so it catches a broken image, not a difference between beamformers."""
    cnr = [ours_f175[n][0]["cnr_mean_db"] for n in (1, 11, 75)]
    assert cnr[0] < cnr[1] < cnr[2]                      # compounding raises contrast
    for n in (1, 11, 75):
        contrast, widths = ours_f175[n]
        ref = bm.PICMUS_REFERENCE_DAS_SIMU[n]
        assert ref["cnr_db"] - 1.5 < contrast["cnr_mean_db"] <= ref["cnr_db"], (n, contrast["cnr_mean_db"])
        assert widths["axial_mm"] == pytest.approx(ref["axial_mm"], abs=0.02), n      # sanity check (docstring)
        assert 0.85 * ref["lateral_mm"] < widths["lateral_mm"] <= ref["lateral_mm"], (n, widths["lateral_mm"])
