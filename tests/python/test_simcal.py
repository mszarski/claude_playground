"""Calibrating the simulator to a log: a log is made with hidden settings, and the
calibration must find them from the log alone."""

import copy

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon, simcal
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

HOST = gs.core.Backend.Host
RANGES = np.arange(25.0, 295.0, 0.06)[:4320]
BEARINGS = np.radians(np.arange(-40.0, 40.01, 1.0))
# What the "real" sea differs by from the scenario the calibration starts from.
TRUE_BOTTOM_DB, TRUE_SURFACE_DB, TRUE_NOISE_DB = 6.0, -4.0, 38.0


def start_cfg():
    cfg = load_scenario_config("fls_120k_transit")
    cfg["environment"]["water_depth"] = 45.0                     # shallower, so the seabed is well inside the beam
    cfg["propagation"]["bottom_z"] = 45.0
    cfg["reverberation"][0]["boundary_z"] = 45.0
    cfg["fidelity"] = "free_field"
    cfg.pop("propagation")
    return cfg


@pytest.fixture(scope="module")
def log():
    """Six pings of a sea whose bottom is 6 dB stronger, surface 4 dB weaker and noise quite different,
    with the scenario's targets in it."""
    cfg = start_cfg()
    cfg["reverberation"][0]["offset_db"] = TRUE_BOTTOM_DB
    cfg["reverberation"][1]["offset_db"] = TRUE_SURFACE_DB
    cfg["receiver"]["ambient_noise"] = {"level_db": TRUE_NOISE_DB}
    cfg["pings"]["count"] = 6
    cfg["seed"] = 99
    return run_scenario(cfg, backend=HOST).records


def test_absolute_log_gives_back_the_hidden_strengths_and_noise(log):
    obs = simcal.observe_records(log, RANGES, BEARINGS)
    cal = simcal.calibrate_fls(obs, start_cfg())
    print(cal.report())
    bottom, surface = cal.boundaries
    assert bottom["constrained"] and bottom["offset_db"] == pytest.approx(TRUE_BOTTOM_DB, abs=1.0)
    assert surface["constrained"] and surface["offset_db"] == pytest.approx(TRUE_SURFACE_DB, abs=1.5)
    assert cal.noise_constrained and cal.reference_noise_level_db + cal.noise_offset_db == pytest.approx(TRUE_NOISE_DB, abs=1.0)
    assert cal.fit_rms_db < 1.0
    # Applying the result makes the simulator reproduce the log's background.
    cfg = start_cfg()
    over = cal.overrides()
    cfg["reverberation"], cfg["receiver"] = over["reverberation"], dict(cfg["receiver"], **over["receiver"])
    cfg["scene"], cfg["pings"]["count"] = [], 3
    again = simcal.observe_records(run_scenario(cfg, backend=HOST).records, RANGES, BEARINGS)
    assert np.sqrt(np.mean((10 * np.log10(again.profile / obs.profile)) ** 2)) < 1.0


def test_relative_log_gives_differences_only(log):
    obs = simcal.observe_records(log, RANGES, BEARINGS)
    scaled = simcal.FlsObservation(obs.ranges, obs.profile * 10 ** 3.7, obs.bearings, "relative", obs.pings)   # +37 dB, unknown
    cal = simcal.calibrate_fls(scaled, start_cfg(), anchor="noise")
    bottom, surface = cal.boundaries
    assert cal.anchor == "noise"
    # Differences between terms survive the unknown gain.
    assert bottom["offset_db"] - surface["offset_db"] == pytest.approx(TRUE_BOTTOM_DB - TRUE_SURFACE_DB, abs=1.5)
    noise = cal.reference_noise_level_db + cal.noise_offset_db
    scenario_noise = float(simcal.calibrate_fls.__globals__["np"].asarray(0.0))            # placeholder for clarity below
    from gpusonar.sim.noise import wenz_spectrum_db
    scenario_noise = float(wenz_spectrum_db(np.array([120e3]), 5.0, 0.5)[0])
    assert noise == pytest.approx(scenario_noise, abs=0.01)       # held where the scenario has it
    assert bottom["offset_db"] - (noise - TRUE_NOISE_DB) == pytest.approx(TRUE_BOTTOM_DB, abs=1.0)
    by_boundary = simcal.calibrate_fls(scaled, start_cfg(), anchor="boundary:0")
    assert by_boundary.boundaries[0]["offset_db"] == pytest.approx(0.0, abs=1e-9)


def test_a_term_the_log_cannot_see_is_reported_not_guessed():
    cfg = start_cfg()
    cfg["receiver"]["ambient_noise"] = {"level_db": 5.0}          # far below the reverberation everywhere
    cfg["pings"]["count"] = 3
    obs = simcal.observe_records(run_scenario(cfg, backend=HOST).records, RANGES, BEARINGS)
    cal = simcal.calibrate_fls(obs, start_cfg())
    assert not cal.noise_constrained and "not constrained" in cal.report()
    assert "receiver" not in cal.overrides()                      # the scenario's noise is left alone
    assert cal.boundaries[0]["offset_db"] == pytest.approx(0.0, abs=1.0)


def test_frames_from_another_beamformer_are_relative(log):
    frames = []
    for rec in log[:3]:
        img = recon.fan_image([rec.copy()], RANGES, BEARINGS, interp="linear", window="hann", backend=HOST)
        frames.append(np.abs(img.image) ** 2 * 123.0)
    obs = simcal.observe_frames(np.array(frames), RANGES, BEARINGS)
    assert obs.levels == "relative" and obs.pings == 3
    cal = simcal.calibrate_fls(obs, start_cfg())
    assert cal.boundaries[0]["offset_db"] - cal.boundaries[1]["offset_db"] == pytest.approx(TRUE_BOTTOM_DB - TRUE_SURFACE_DB, abs=2.0)


# --- SAS images ------------------------------------------------------------------------------------

def sas_image(density, seed=5):
    cfg = load_scenario_config("sas_minsas_speckle")
    cfg["scene"] = [{"type": "seafloor", "label": "seabed", "x_range": [0.0, 4.0], "y_range": [48.5, 51.5], "density": density,
                     "lambert_mu_db": -27.0},
                    {"type": "box", "label": "block", "min": [1.6, 49.0, -0.4], "max": [2.2, 49.4, 0.0], "density": 4000.0, "mu_db": -8.0}]
    cfg["platform"]["legs"][0].update(start=[-3.0, 0.0, -12.0], pings=20)
    cfg["seed"] = seed
    records = run_scenario(cfg, backend=HOST).records
    x, y = np.arange(0.3, 3.7, 0.02), np.arange(48.7, 51.3, 0.02)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.abs(recon.sas_image(records, np.stack([X, Y, np.zeros_like(X)], axis=-1), interp="linear", backend=HOST)) ** 2


@pytest.fixture(scope="module")
def sas_images():
    return {d: sas_image(d) for d in (400.0, 6000.0)}


def test_speckle_statistics_tell_sparse_from_developed(sas_images):
    sparse = simcal.calibrate_sas_image(sas_images[400.0], 0.02, block=32)
    dense = simcal.calibrate_sas_image(sas_images[6000.0], 0.02, block=32)
    print(sparse.report(), dense.report(), sep="\n")
    assert dense.scintillation_index < 1.6 < sparse.scintillation_index
    assert 2e-4 < dense.cell_area_m2 < 12e-4                      # the design cell is about 5 cm^2
    # Sparse seabed: the density comes back within a factor of two and a half.
    assert 400.0 / 2.5 < sparse.density_per_m2 < 400.0 * 2.5
    # Developed speckle only says "at least a few per cell".
    assert dense.scatterers_per_cell > 2.0 and dense.density_per_m2 > 2000.0
    # Both images hold a block with a shadow behind it.
    assert dense.shadow_below_seabed_db < -6.0


def test_an_overall_gain_changes_nothing_for_an_image(sas_images):
    a = simcal.calibrate_sas_image(sas_images[6000.0], 0.02, block=32)
    b = simcal.calibrate_sas_image(sas_images[6000.0] * 1e7, 0.02, block=32)
    assert b.scintillation_index == pytest.approx(a.scintillation_index) and b.shadow_below_seabed_db == pytest.approx(a.shadow_below_seabed_db)


def test_display_scale_can_be_undone():
    power = np.random.default_rng(0).exponential(1.0, (64, 64)) + 1e-3
    db = np.clip(10 * np.log10(power), -20.0, 10.0)
    grey = np.round((db + 20.0) / 30.0 * 255.0)
    back = simcal.from_display_db(grey, -20.0, 10.0)
    inside = (db > -19.9) & (db < 9.9)
    assert np.allclose(10 * np.log10(back[inside]), db[inside], atol=0.07)
