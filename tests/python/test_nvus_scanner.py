"""Milestone 11: the ultrasound source driven through the NVIDIA reference's own native
runtime (holoscan-ultrasound-imaging at 24d405e), with its simulated hardware.

Needs the reference's `nvus` package with its native module built; set
GPUSONAR_NVUS_PYTHON to the directory holding it. Skipped otherwise.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import NvusScannerSource, ReferenceUnavailable, make_source, nvus
from gpusonar.sources import nvus_scanner

try:
    native = nvus_scanner.load_reference()
except ReferenceUnavailable as exc:
    pytest.skip(f"reference not built: {exc}", allow_module_level=True)


PATCHED = nvus_scanner.max_depth(native) > 1.0      # built with this repository's long-window patch


def test_the_compiler_limit_is_found_and_reported():
    limit = nvus_scanner.max_depth(native)
    if PATCHED:
        assert 3.4 < limit < 3.6                     # 262,144 samples at 50 MHz, less the 10% margin
        with pytest.raises(ValueError, match="limit is 3.5"):
            NvusScannerSource(depth_m=5.0)
    else:
        assert 0.85 < limit < 0.97                   # upstream: 65,536 samples
        with pytest.raises(ValueError, match="limit is 0.88"):
            NvusScannerSource(depth_m=2.5)           # the 2 to 3 m pool window does not compile


@pytest.mark.skipif(not PATCHED, reason="reference built without the long-window patch")
def test_pool_window_compiles_and_runs_on_the_fast_path():
    """The plan's 2 to 3 m window through the reference's native compiler and loop."""
    source = NvusScannerSource(depth_m=2.95)
    assert [e[1] for e in source.events] == ["odd", "even"]
    assert source.events[0][2] == source.events[1][2] == 219258          # 4.4 ms at 50 MHz
    records = list(source)
    assert len(records) == 2 and source.incomplete_observations == 0
    s = records[0].samples
    assert s.array.shape == (64, 219258)
    assert 1480.0 * (s.start_time + s.n_samples / s.sample_rate) / 2 > 3.2    # reaches past 3 m


@pytest.mark.skipif(not PATCHED, reason="reference built without the long-window patch")
def test_far_wall_echo_through_the_patched_fast_path():
    target = np.array([2.5, 0.0, 0.0])
    sonar = SonarModel.load("l12_5d", {"receive": {"window": {"range": [2.4, 2.6]}},
                                       "transmit": {"waveform": {"amplitude": 5000.0}}})
    sim = [simulate_ping(sonar, make_pose(), Scene().add_points([target]), environment={"sound_speed": 1480.0},
                         absorption_model="none", sub_sequence=i, sequence=i, adc=False) for i in range(2)]

    def hook(index, rx_select, data):
        s = sim[0 if rx_select == "odd" else 1].samples
        counts = np.zeros((64, len(data) // nvus.BYTES_PER_SAMPLE), dtype=np.int16)
        first = int(round((s.start_time - source.start_time) * s.sample_rate))
        counts[:, first:first + s.n_samples] = np.clip(np.rint(s.array * 2048.0), -2048, 2047)
        return nvus.encode_frame(counts)

    source = NvusScannerSource(depth_m=2.95, frame_hook=hook)
    records = list(source)
    ranges = 2.5 + np.arange(-0.004, 0.0041, 0.0001)
    bearings = np.radians(np.arange(-1.0, 1.01, 0.05))
    img = np.abs(recon.fan_image(records, ranges, bearings, interp="sinc").image)
    ir, ib = np.unravel_index(np.argmax(img), img.shape)
    assert ranges[ir] == pytest.approx(2.5, abs=4e-4) and abs(np.degrees(bearings[ib])) < 0.15


def test_scan_is_one_plane_wave_per_mux_half_at_the_longest_window():
    source = NvusScannerSource(depth_m=0.88)
    assert [e[1] for e in source.events] == ["odd", "even"]
    assert 65300 < source.events[0][2] == source.events[1][2] <= 65536
    assert source.manifest["sample_rate_hz"] == 50_000_000
    assert source.start_time == pytest.approx((80 - 2020) / 200e6)       # the window opens before the transmit
    assert source.simulated


def test_simulated_hardware_run_yields_valid_records():
    source = make_source({"type": "nvus_scanner", "repetitions": 3, "depth_m": 0.88})
    records = list(source)
    assert len(records) == 6 and source.incomplete_observations == 0
    assert source.hardware.program_calls == 1 and source.hardware.stop_calls == 1
    for i, r in enumerate(records):
        assert gs.validate(r).error_count == 0, str(gs.validate(r))
        assert r.metadata.sub_sequence == i % 2 and r.metadata.sequence == i
        assert r.samples.array.shape == (64, source.events[0][2])
        assert np.abs(r.samples.array).max() <= 1.0                       # placeholder content, inside the ADC range
        assert r.transmit.kind == gs.TransmitKind.PlaneWave
        extra = json.loads(r.metadata.extra_json)
        assert extra["simulated_hardware"] and extra["reference_commit"] == "24d405e"
        assert extra["receive_attenuation_code"] == 23
    first = np.array(records[0].rx_positions())
    second = np.array(records[1].rx_positions())
    assert first[:, 1].max() < second[:, 1].min()                        # elements 0-63, then 64-127
    assert records[2].transmit.time_ns > records[0].transmit.time_ns


def test_echoes_through_the_reference_loop_image_at_the_target():
    """Simulated echoes are put into the frames the reference's loop hands over; the image
    from the resulting records has the target where it is."""
    target = np.array([0.6, 0.0, 0.0])                                    # on the boresight
    sonar = SonarModel.load("l12_5d", {"receive": {"window": {"range": [0.5, 0.7]}},
                                       "transmit": {"waveform": {"amplitude": 500.0}}})
    sim = [simulate_ping(sonar, make_pose(), Scene().add_points([target]), environment={"sound_speed": 1480.0},
                         absorption_model="none", sub_sequence=i, sequence=i, adc=False) for i in range(2)]

    def hook(index, rx_select, data):
        s = sim[0 if rx_select == "odd" else 1].samples
        counts = np.zeros((64, len(data) // nvus.BYTES_PER_SAMPLE), dtype=np.int16)
        first = int(round((s.start_time - source.start_time) * s.sample_rate))
        counts[:, first:first + s.n_samples] = np.clip(np.rint(s.array * 2048.0), -2048, 2047)
        return nvus.encode_frame(counts)

    source = NvusScannerSource(depth_m=0.88, frame_hook=hook)
    records = list(source)
    assert len(records) == 2 and all(np.abs(r.samples.array).max() > 0.01 for r in records)
    ranges = 0.6 + np.arange(-0.004, 0.0041, 0.0001)
    bearings = np.radians(np.arange(-2.0, 2.01, 0.05))
    img = np.abs(recon.fan_image(records, ranges, bearings, interp="sinc").image)
    ir, ib = np.unravel_index(np.argmax(img), img.shape)
    assert ranges[ir] == pytest.approx(0.6, abs=4e-4)                     # within about a wavelength
    assert abs(np.degrees(bearings[ib])) < 0.15
