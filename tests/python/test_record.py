"""Record container from Python: round trip, validator, and plain-h5py readability."""

import h5py
import numpy as np

import gpusonar as gs
from conftest import make_record


def test_round_trip(tmp_path, record):
    path = str(tmp_path / "a.h5")
    opts = gs.WriterOptions()
    opts.config_yaml = "sonar: test\n"
    with gs.RecordWriter(path, opts) as w:
        assert w.write(record) == 0
        w.write_truth("points", np.arange(6.0).reshape(2, 3))
    reader = gs.RecordReader(path)
    assert len(reader) == 1
    assert reader.info.schema_version == gs.core.SCHEMA_VERSION
    assert reader.info.config_yaml == "sonar: test\n"
    back = reader.read(0)
    np.testing.assert_array_equal(back.samples.array, record.samples.array)
    assert back.samples.array.dtype == np.complex64
    assert back.metadata.array_id == "test"
    assert back.geometry.rx[3].lever_arm == record.geometry.rx[3].lever_arm
    np.testing.assert_array_equal(reader.read_truth("points"), np.arange(6.0).reshape(2, 3))
    assert gs.validate_file(path).ok


def test_real_samples_round_trip(tmp_path):
    rec = make_record(complex_data=False)
    path = str(tmp_path / "r.h5")
    with gs.RecordWriter(path) as w:
        w.write(rec)
    back = gs.RecordReader(path).read(0)
    assert back.samples.kind == gs.SampleKind.RealPassband
    assert back.samples.array.dtype == np.float32
    np.testing.assert_array_equal(back.samples.array, rec.samples.array)


def test_file_is_readable_with_plain_h5py(tmp_path, record):
    """The container is documented HDF5: no gpusonar code is needed to read it."""
    path = str(tmp_path / "b.h5")
    with gs.RecordWriter(path) as w:
        w.write(record)
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_name"] == "gpusonar.record"
        ping = f["pings/00000000"]
        samples = ping["samples"][...]
        assert samples.dtype == np.complex64
        np.testing.assert_array_equal(samples, record.samples.array)
        assert ping.attrs["sample_rate"] == 64000.0
        assert ping["geometry/rx_lever_arm"].shape == (4, 3)
        assert ping["transmit"].attrs["f_stop"] == 10000.0


def test_validator_reports_field_paths(record):
    assert gs.validate(record).ok
    env = record.environment
    env.sound_speed = -5.0
    record.environment = env
    report = gs.validate(record)
    assert not report.ok
    assert any(i.field == "environment.sound_speed" for i in report.issues)


def test_rx_positions_follow_pose(record):
    geo = record.geometry
    pose = geo.rx_pose
    pose.position = (10.0, 0.0, -5.0)
    pose.rpy = (0.0, 0.0, np.pi / 2)  # heading east: the along-body array now runs east
    pose.velocity = (0.0, 1.0, 0.0)
    geo.rx_pose = pose
    record.geometry = geo
    p0 = record.rx_positions()
    np.testing.assert_allclose(p0[:, 0], 10.0, atol=1e-12)
    np.testing.assert_allclose(p0[:, 1], 0.094 * np.arange(4), atol=1e-12)
    p1 = record.rx_positions(2.0)
    np.testing.assert_allclose(p1[:, 1] - p0[:, 1], 2.0, atol=1e-12)
