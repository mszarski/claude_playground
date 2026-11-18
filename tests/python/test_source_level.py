"""Source levels are RMS levels (dB re 1 uPa at 1 m), everywhere, and the sqrt(2) to a peak amplitude is applied in
one place (gpusonar/sonar.py): the simulated echo, the sonar equation and the overload figures agree."""

import warnings

import numpy as np
import pytest

from gpusonar import buildspec as bs
from gpusonar import sonar as sonar_module
from gpusonar.config import load_sonar_config
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import PEAK_OVER_RMS_DB, SonarModel, peak_pressure_at_1m, source_level_db_rms
from gpusonar.sonar_equation import SonarEquation

CW = {"transmit": {"waveform": {"type": "cw", "f_start": 5000.0, "f_stop": 5000.0, "duration": 0.02, "window_alpha": 0.0}},
      "receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1100.0]}}}


def test_the_one_place():
    assert PEAK_OVER_RMS_DB == pytest.approx(3.0103, abs=1e-4)
    assert peak_pressure_at_1m(200.0) == pytest.approx(np.sqrt(2.0) * 1e10)
    sonar = SonarModel.load("popoto_long_range")
    rx = sonar.config["receive"]
    assert sonar.source_level_db_rms == 197.0
    assert sonar.amplitude_at_1m() == pytest.approx(np.sqrt(2.0) * 10.0 ** ((197.0 + rx["sensitivity_db"] + rx.get("gain_db", 0.0)) / 20.0))
    record = sonar.make_record(make_pose(), environment={"sound_speed": 1500.0})
    # a record holds PEAK values (core/record.hpp): the level of the peak pressure, and the peak amplitude it gives
    w = record.transmit.waveform
    assert w.source_level_db == pytest.approx(197.0 + PEAK_OVER_RMS_DB) and sonar_module.record_source_level_db_rms(record) == pytest.approx(197.0)
    assert w.amplitude == pytest.approx(sonar.amplitude_at_1m()) == pytest.approx(10.0 ** ((w.source_level_db + rx["sensitivity_db"] + rx.get("gain_db", 0.0)) / 20.0))
    # an uncalibrated sonar: `amplitude` is a peak value and nothing is scaled
    plain = SonarModel.load("popoto_long_range", {"transmit": {"waveform": {"source_level_db_rms": None, "amplitude": 2.5}}})
    assert plain.source_level_db_rms is None and plain.amplitude_at_1m() == 2.5


def test_the_old_key_is_read_as_rms_with_a_warning():
    with pytest.warns(FutureWarning, match="source_level_db_rms"):
        old = SonarModel.load("popoto_long_range", {"transmit": {"waveform": {"source_level_db": 190.0}}})
    assert old.source_level_db_rms == 190.0                                    # it replaced the base file's 197 and is RMS
    new = SonarModel.load("popoto_long_range", {"transmit": {"waveform": {"source_level_db_rms": 190.0}}})
    assert old.amplitude_at_1m() == new.amplitude_at_1m()
    with pytest.raises(ValueError, match="both given and differ"):
        source_level_db_rms({"source_level_db": 190.0, "source_level_db_rms": 197.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("popoto_long_range", "flank_32", "fls_120k", "sas_minsas120", "popoto_8", "popoto_face_1row"):
            cfg = load_sonar_config(name)                                       # no shipped file uses the old key
            assert "source_level_db" not in cfg["transmit"]["waveform"] and cfg["transmit"]["waveform"].get("source_level_db_rms") is not None


def test_simulated_echo_is_the_sonar_equations_level():
    """A 0 dB point target at 1 km, no absorption, a rectangular tone: the echo's RMS pressure at the hydrophone is
    SL - 2 TL + TS. In a baseband record the envelope is the peak, sqrt(2) above the RMS."""
    sonar = SonarModel.load("popoto_long_range", CW)
    rx = sonar.config["receive"]
    scene = Scene().add_points([[0.0, 1000.0, 0.0]], target_strength_db=0.0)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")
    envelope = float(np.median(np.sort(np.abs(rec.samples.array[0]))[-100:]))       # the plateau of the pulse
    volts_per_upa = 10.0 ** ((rx["sensitivity_db"] + rx.get("gain_db", 0.0)) / 20.0)
    rms_level = 20.0 * np.log10(envelope / np.sqrt(2.0) / volts_per_upa)
    assert rms_level == pytest.approx(197.0 - 2.0 * 20.0 * np.log10(1000.0) + 0.0, abs=0.05)
    eq = SonarEquation.from_sonar(sonar, noise_spectrum_db=50.0, absorption_model="none")
    assert eq.source_level_db == 197.0 and float(eq.echo_level(1000.0)) == pytest.approx(rms_level, abs=0.05)
    # and the power: a real record's mean square over the pulse is the RMS pressure squared
    real = SonarModel.load("popoto_long_range", {**CW, "receive": {"kind": "real", "sample_rate": 32000.0, "window": {"range": [990.0, 1005.0]}}})
    x = simulate_ping(real, make_pose(), scene, environment={"sound_speed": 1500.0}, adc=False, absorption_model="none").samples.array[0]
    on = np.abs(x) > 0.05 * np.abs(x).max()
    core_ = x[np.flatnonzero(on)[0] + 40:np.flatnonzero(on)[-1] - 40]
    assert 10.0 * np.log10(np.mean(core_ ** 2) / volts_per_upa ** 2) == pytest.approx(197.0 - 120.0, abs=0.1)


def test_direct_sound_overload_is_a_peak_against_a_peak():
    d = bs.derive(bs.load("popoto_testauv_1x32"))
    dr = d["direct"]
    assert dr["level_db"] == pytest.approx(d["source_level_db"] - 20.0 * np.log10(dr["nearest_m"]))          # an RMS level
    peak_volts = 10.0 ** ((dr["level_db"] + PEAK_OVER_RMS_DB + d["sensitivity_db"] + d["gain_db"]) / 20.0)
    assert dr["volts"] == pytest.approx(peak_volts) and dr["over_full_scale_db"] == pytest.approx(20.0 * np.log10(peak_volts / d["adc_full_scale_v"]))
    # the simulator's direct arrival is that same peak
    model = SonarModel.load("popoto_testauv_1row")
    assert model.amplitude_at_1m() / dr["nearest_m"] == pytest.approx(dr["volts"], rel=1e-6)
    assert sonar_module.SOURCE_LEVEL_KEY == "source_level_db_rms"
