"""Milestone 12: the long-range scenarios at the `eigenray` tier, against the simple
spherical-spreading estimate. Needs Bellhop (external; GPUSONAR_BELLHOP)."""

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar import flank
from gpusonar.config import load_scenario_config
from gpusonar.sim import find_bellhop, run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.sonar_equation import SonarEquation

pytestmark = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")


def output_snr_db(record, target, bearings_deg=(-1.0, 0.0, 1.0)):
    """Beamform at the target's bearing; peak in the echo's arrival window over the mean
    noise power elsewhere on the same beam."""
    bm = flank.farfield_beamform(record, np.radians(bearings_deg)).valid()
    row = np.abs(bm.data[1])
    tx = np.array(record.tx_position())
    centre = record.rx_positions().mean(axis=0)
    tau = (np.linalg.norm(target - tx) + np.linalg.norm(target - centre)) / record.environment.sound_speed
    fs = 1.0 / (bm.times[1] - bm.times[0])
    i = int(round((tau - bm.times[0]) * fs))
    spread = int(0.06 * fs)                               # multipath arrives up to tens of ms late
    peak = row[max(i - 5, 0):i + spread].max()
    quiet = np.r_[0:max(i - int(0.2 * fs), 1), min(i + int(0.25 * fs), len(row) - 1):len(row)]
    return 10 * np.log10(peak**2 / np.mean(row[quiet] ** 2))


def scenario_snr(name):
    cfg = load_scenario_config(name)
    # The 10 km case has thousands of two-way paths; synthesis runs on the GPU when there is one.
    backend = core.Backend.Cuda if core.cuda_available() else core.Backend.Host
    result = run_scenario(cfg, backend=backend)
    assert cfg["fidelity"] == "eigenray"
    return output_snr_db(result.records[0], result.scene.positions[0]), cfg


@pytest.mark.parametrize("name, range_m", [("flank_target_1km", 1000.0), ("flank_target_5km", 5000.0),
                                           ("flank_target_10km", 10000.0)])
def test_flank_target_at_the_eigenray_tier(name, range_m, capsys):
    snr, cfg = scenario_snr(name)
    sonar = SonarModel.load("flank_32")
    eq = SonarEquation.from_sonar(sonar, 45.0, target_strength_db=10.0, temperature=12.0, salinity=35.0,
                                  absorption_model="ainslie_mccolm")
    spherical = eq.snr(range_m)
    with capsys.disabled():
        print(f"\n[eigenray tier] {name}: SNR {snr:.1f} dB; spherical-spreading estimate {spherical:.1f} dB; "
              f"difference {snr - spherical:+.1f} dB")
    # Shallow-water ducting keeps more energy than spherical spreading predicts; the gap
    # opens with range. Multipath also splits the echo, so the peak gains less than the
    # energy does. The scenarios include bottom reverberation (scattering-function
    # method). Bounds are loose: this is a comparison, not a calibration.
    if range_m >= 10000.0:
        # Noise-limited: bottom reverberation is below the ambient noise here.
        assert spherical + 3.0 < snr < spherical + 30.0
    elif range_m >= 5000.0:
        # Reverberation and noise are comparable; ducting gain and reverberation roughly cancel.
        assert spherical - 6.0 < snr < spherical + 6.0
    else:
        # Reverberation-limited: the spherical estimate in noise alone is far too optimistic.
        assert 15.0 < snr < spherical - 10.0


@pytest.mark.parametrize("name, range_m", [("popoto_trial_100m", 100.0), ("popoto_trial_1000m", 1000.0),
                                           ("popoto_trial_2000m", 2000.0)])
def test_popoto_trial_at_the_eigenray_tier(name, range_m, capsys):
    snr, cfg = scenario_snr(name)
    sonar = SonarModel.load("popoto_8")
    eq = SonarEquation.from_sonar(sonar, 38.0, target_strength_db=10.0, temperature=12.0, salinity=35.0,
                                  absorption_model="ainslie_mccolm")
    spherical = eq.snr(range_m)
    with capsys.disabled():
        print(f"\n[eigenray tier] {name}: SNR {snr:.1f} dB; spherical-spreading estimate {spherical:.1f} dB; "
              f"difference {snr - spherical:+.1f} dB")
    if range_m >= 2000.0:
        assert spherical + 3.0 < snr < spherical + 30.0          # noise-limited, ducted
    else:
        assert 15.0 < snr < spherical - 3.0                      # reverberation-limited
