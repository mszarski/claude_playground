"""The adapter from the Holoscan ultrasound reference's capture frames to ping records.

No hardware here. The decoder is checked against the reference's own CPU decoder when a
checkout of the reference is available (GPUSONAR_NVUS_DIR, or the default clone path),
and against the documented layout otherwise.
"""

import importlib.util
import os
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import nvus

REFERENCE = pathlib.Path(os.environ.get("GPUSONAR_NVUS_DIR", "/home/user/nvidia-holoscan/holoscan-ultrasound-imaging"))


def test_documented_layout_one_sample():
    """Physical channel c of an AFE rides lane c >> 1 of the record with parity c & 1."""
    samples = np.zeros((64, 1), dtype=np.int16)
    samples[5, 0] = 0x123          # AFE 0, channel 5: lane 2, second record
    samples[32 + 30, 0] = -7       # AFE 1, channel 30: lane 15, first record
    frame = nvus.encode_frame(samples)
    assert len(frame) == 96
    record0, record1 = frame[:48], frame[48:]
    word = int.from_bytes(record1[:24], "little")                  # second record, AFE 0
    assert (word >> (12 * 2)) & 0xFFF == 0x123
    word = int.from_bytes(record0[24:48], "little")                # first record, AFE 1
    assert (word >> (12 * 15)) & 0xFFF == (-7) & 0xFFF
    np.testing.assert_array_equal(nvus.decode_frame(frame), samples)


def test_round_trip_and_partial_tail():
    rng = np.random.default_rng(0)
    samples = rng.integers(-2048, 2048, (64, 300), dtype=np.int16)
    frame = nvus.encode_frame(samples)
    np.testing.assert_array_equal(nvus.decode_frame(frame), samples)
    np.testing.assert_array_equal(nvus.decode_frame(frame + b"\x00" * 50), samples)   # incomplete sample dropped
    with pytest.raises(ValueError):
        nvus.encode_frame(np.full((64, 1), 5000))


@pytest.mark.skipif(not (REFERENCE / "python/nvus/raw.py").exists(), reason="reference checkout not available")
def test_decoder_matches_the_references_own_decoder():
    spec = importlib.util.spec_from_file_location("nvus_raw", REFERENCE / "python/nvus/raw.py")
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, 96 * 64, dtype=np.uint8).tobytes()       # arbitrary bytes
    ours = nvus.decode_frame(frame)
    # The reference's scalar helpers define the format: check a handful of samples through them.
    for sample in (0, 17, 63):
        for column in (0, 1, 31, 32, 45, 63):
            afe, channel = divmod(column, 32)
            record = frame[(2 * sample + (channel & 1)) * 48:][:48]
            lanes = raw.unpack_afe_sample(record[afe * 24:(afe + 1) * 24])
            assert ours[column, sample] == raw.sign_extend_12(lanes[channel >> 1])
    assert raw.element_indices("odd") == list(range(64))
    assert raw.element_indices("even") == list(range(64, 128))
    assert (raw.RECORD_BYTES, raw.FS, raw.N_ELEMENTS, raw.PITCH_M) == (48, 50e6, 128, 0.3e-3)


def test_captured_frames_image_like_the_simulated_pings():
    """End to end without hardware: simulated pings are packed into the reference's frame
    format, decoded by the adapter, and imaged. The picture must match imaging the
    simulated records directly."""
    sonar = SonarModel.load("l12_5d", {"receive": {"window": {"range": [1.45, 1.55]}}})
    scene = Scene().add_points([[1.5, 0.04, 0.0]], amplitude=2.0)
    sim = [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1480.0},
                         absorption_model="none", sub_sequence=i, sequence=i) for i in range(2)]
    records = []
    for rec, select in zip(sim, ("odd", "even")):
        counts = np.rint(rec.samples.array * 2048.0).astype(np.int16)        # ADC counts, as the FPGA sends them
        assert np.abs(counts).max() > 100
        frame = nvus.encode_frame(counts)
        out = nvus.frame_to_record(frame, sonar, select, start_time=rec.samples.start_time,
                                   time_ns=rec.transmit.time_ns, zero_capture_spike=False)
        assert gs.validate(out).ok, str(gs.validate(out))
        np.testing.assert_allclose(out.samples.array, rec.samples.array, atol=1e-7)
        records.append(out)
    assert records[0].metadata.channel_element == list(range(64))
    assert records[1].metadata.channel_element == list(range(64, 128))
    ranges = np.linspace(1.48, 1.52, 81)
    bearings = np.radians(np.linspace(-2.0, 5.0, 141))
    a = np.abs(recon.fan_image(records, ranges, bearings).image)
    b = np.abs(recon.fan_image(sim, ranges, bearings).image)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6 * b.max())
    ir, ib = np.unravel_index(np.argmax(a), a.shape)
    assert np.degrees(bearings[ib]) == pytest.approx(np.degrees(np.arctan2(0.04, 1.5)), abs=0.1)


def test_capture_spike_is_zeroed_and_wrong_mux_is_rejected():
    sonar = SonarModel.load("l12_5d")
    frame = nvus.encode_frame(np.full((64, 32), 100, dtype=np.int16))
    rec = nvus.frame_to_record(frame, sonar, "odd")
    x = rec.samples.array
    assert np.all(x[:, :8] == 0) and np.all(x[:, 8:] > 0)
    with pytest.raises(ValueError):
        nvus.frame_to_record(frame, sonar, "both")
    # The 64-element phased array fills the 64 channels directly (the plan drops the mux
    # for it): the first setting is valid, a second half does not exist.
    assert nvus.frame_to_record(frame, SonarModel.load("p4_2v"), "odd").samples.n_channels == 64
    with pytest.raises(ValueError, match="mux halves"):
        nvus.frame_to_record(frame, SonarModel.load("p4_2v"), "even")


# --- the reference rig's receive attenuator ramp ---------------------------------------------

def test_dtgc_model_reproduces_the_reference_register_values():
    """Values worked out from the reference's encoder (HOLOSCAN_NOTES.md section 6)."""
    from gpusonar import hardware
    shipped = hardware.tx73l64_dtgc(12.0, 2.0, 10.0)
    assert shipped["reg_0x05"] == 0x3230 and shipped["reg_0x06"] == 0x3015
    assert shipped["ramp_s"] == pytest.approx(10.08e-6)
    assert shipped["tvg"] == {"time": [pytest.approx(2e-6), pytest.approx(12.08e-6)], "gain_db": [-12.0, 0.0]}
    # The longest ramp one segment can hold at 12 dB.
    assert hardware.tx73l64_dtgc(12.0, 2.0, 122.4)["ramp_s"] == pytest.approx(122.4e-6)
    for bad in ((12.0, 2.0, 130.0), (12.5, 2.0, 10.0), (12.0, 11.0, 10.0), (0.1, 2.0, 10.0)):
        with pytest.raises(ValueError):
            hardware.tx73l64_dtgc(*bad)
    assert hardware.tx73l64_fixed_attenuation_db(23) == -5.75
    with pytest.raises(ValueError):
        hardware.tx73l64_fixed_attenuation_db(49)


def test_reference_probe_config_declares_the_ramp_and_it_is_over_before_the_window():
    from gpusonar.sim import make_pose
    sonar = SonarModel.load("l12_5d")
    rec = sonar.make_record(make_pose())
    assert list(rec.metadata.tvg_gain_db) == [-12.0, 0.0]
    assert rec.metadata.tvg_time[1] == pytest.approx(12.08e-6)
    assert rec.samples.start_time > 50 * rec.metadata.tvg_time[1]        # the window opens about 1 ms in
    with pytest.raises(ValueError, match="not both"):
        SonarModel.load("l12_5d", {"receive": {"tvg": {"time": [0.0, 1e-3], "gain_db": [0.0, 6.0]}}}).make_record(make_pose())
