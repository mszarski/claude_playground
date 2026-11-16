// Record container: round trip, integer storage, schema checks, validator.
#include <gtest/gtest.h>

#include <hdf5.h>

#include <cmath>
#include <cstdio>

#include "sonar/io/record_file.hpp"
#include "sonar/io/validate.hpp"
#include "test_support.hpp"

using namespace sonar;
using sonar::test::make_full_record;
using sonar::test::temp_path;

namespace {

void expect_vec(const Vec3d& a, const Vec3d& b) {
  auto same = [](double x, double y) { return (std::isnan(x) && std::isnan(y)) || x == y; };
  EXPECT_TRUE(same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z));
}
void expect_pose(const Pose& a, const Pose& b) {
  expect_vec(a.position, b.position);
  expect_vec(a.rpy, b.rpy);
  expect_vec(a.velocity, b.velocity);
  expect_vec(a.rates, b.rates);
}
bool same(double x, double y) { return (std::isnan(x) && std::isnan(y)) || x == y; }

void expect_equal(const PingRecord& a, const PingRecord& b, bool compare_samples = true) {
  EXPECT_EQ(a.samples.kind, b.samples.kind);
  EXPECT_EQ(a.samples.n_channels, b.samples.n_channels);
  EXPECT_EQ(a.samples.n_samples, b.samples.n_samples);
  EXPECT_EQ(a.samples.sample_rate, b.samples.sample_rate);
  EXPECT_EQ(a.samples.center_frequency, b.samples.center_frequency);
  EXPECT_EQ(a.samples.start_time, b.samples.start_time);
  EXPECT_EQ(a.samples.storage, b.samples.storage);
  if (compare_samples) EXPECT_EQ(a.samples.data, b.samples.data);

  const Waveform& wa = a.transmit.waveform;
  const Waveform& wb = b.transmit.waveform;
  EXPECT_EQ(wa.type, wb.type);
  EXPECT_EQ(wa.f_start, wb.f_start);
  EXPECT_EQ(wa.f_stop, wb.f_stop);
  EXPECT_EQ(wa.duration, wb.duration);
  EXPECT_EQ(wa.window_alpha, wb.window_alpha);
  EXPECT_TRUE(same(wa.source_level_db, wb.source_level_db));
  EXPECT_EQ(wa.amplitude, wb.amplitude);
  EXPECT_EQ(wa.replica, wb.replica);
  EXPECT_EQ(a.transmit.time_ns, b.transmit.time_ns);
  EXPECT_EQ(a.transmit.kind, b.transmit.kind);
  expect_vec(a.transmit.direction, b.transmit.direction);

  expect_pose(a.geometry.tx_pose, b.geometry.tx_pose);
  expect_pose(a.geometry.rx_pose, b.geometry.rx_pose);
  expect_vec(a.geometry.tx.lever_arm, b.geometry.tx.lever_arm);
  expect_vec(a.geometry.tx.boresight, b.geometry.tx.boresight);
  EXPECT_TRUE(same(a.geometry.tx.beamwidth_az, b.geometry.tx.beamwidth_az));
  ASSERT_EQ(a.geometry.rx.size(), b.geometry.rx.size());
  for (std::size_t i = 0; i < a.geometry.rx.size(); ++i) {
    expect_vec(a.geometry.rx[i].lever_arm, b.geometry.rx[i].lever_arm);
    expect_vec(a.geometry.rx[i].boresight, b.geometry.rx[i].boresight);
    EXPECT_TRUE(same(a.geometry.rx[i].beamwidth_az, b.geometry.rx[i].beamwidth_az));
    EXPECT_TRUE(same(a.geometry.rx[i].beamwidth_el, b.geometry.rx[i].beamwidth_el));
  }
  expect_vec(a.geometry.frame_origin_lla, b.geometry.frame_origin_lla);

  ASSERT_EQ(a.navigation.ins.size(), b.navigation.ins.size());
  for (std::size_t i = 0; i < a.navigation.ins.size(); ++i) {
    EXPECT_EQ(a.navigation.ins[i].t, b.navigation.ins[i].t);
    expect_vec(a.navigation.ins[i].position, b.navigation.ins[i].position);
    expect_vec(a.navigation.ins[i].rates, b.navigation.ins[i].rates);
  }
  ASSERT_EQ(a.navigation.dvl.size(), b.navigation.dvl.size());
  for (std::size_t i = 0; i < a.navigation.dvl.size(); ++i) {
    EXPECT_EQ(a.navigation.dvl[i].altitude, b.navigation.dvl[i].altitude);
    expect_vec(a.navigation.dvl[i].velocity_body, b.navigation.dvl[i].velocity_body);
  }
  ASSERT_EQ(a.navigation.gps.size(), b.navigation.gps.size());
  ASSERT_EQ(a.navigation.depth.size(), b.navigation.depth.size());
  if (!a.navigation.depth.empty()) EXPECT_EQ(a.navigation.depth[0].depth, b.navigation.depth[0].depth);

  EXPECT_EQ(a.calibration.gain, b.calibration.gain);
  EXPECT_EQ(a.calibration.phase, b.calibration.phase);
  EXPECT_EQ(a.calibration.delay, b.calibration.delay);
  ASSERT_EQ(a.calibration.position_offset.size(), b.calibration.position_offset.size());
  for (std::size_t i = 0; i < a.calibration.position_offset.size(); ++i) {
    expect_vec(a.calibration.position_offset[i], b.calibration.position_offset[i]);
  }

  EXPECT_EQ(a.environment.sound_speed, b.environment.sound_speed);
  EXPECT_TRUE(same(a.environment.temperature, b.environment.temperature));
  EXPECT_TRUE(same(a.environment.salinity, b.environment.salinity));
  EXPECT_TRUE(same(a.environment.water_depth, b.environment.water_depth));
  EXPECT_EQ(a.environment.profile_depth, b.environment.profile_depth);
  EXPECT_EQ(a.environment.profile_speed, b.environment.profile_speed);

  EXPECT_EQ(a.metadata.array_id, b.metadata.array_id);
  EXPECT_EQ(a.metadata.mode, b.metadata.mode);
  EXPECT_EQ(a.metadata.sequence, b.metadata.sequence);
  EXPECT_EQ(a.metadata.sub_sequence, b.metadata.sub_sequence);
  EXPECT_EQ(a.metadata.channel_element, b.metadata.channel_element);
  EXPECT_EQ(a.metadata.channel_board, b.metadata.channel_board);
  ASSERT_EQ(a.metadata.boards.size(), b.metadata.boards.size());
  for (std::size_t i = 0; i < a.metadata.boards.size(); ++i) {
    EXPECT_EQ(a.metadata.boards[i].clock_offset, b.metadata.boards[i].clock_offset);
    EXPECT_EQ(a.metadata.boards[i].clock_drift_ppm, b.metadata.boards[i].clock_drift_ppm);
    EXPECT_TRUE(same(a.metadata.boards[i].pps_sample, b.metadata.boards[i].pps_sample));
    EXPECT_EQ(a.metadata.boards[i].reference_channel, b.metadata.boards[i].reference_channel);
    EXPECT_EQ(a.metadata.boards[i].synchronised, b.metadata.boards[i].synchronised);
  }
  EXPECT_EQ(a.metadata.hardware_gain_db, b.metadata.hardware_gain_db);
  EXPECT_EQ(a.metadata.tvg_time, b.metadata.tvg_time);
  EXPECT_EQ(a.metadata.tvg_gain_db, b.metadata.tvg_gain_db);
  EXPECT_EQ(a.metadata.adc_bits, b.metadata.adc_bits);
  EXPECT_TRUE(same(a.metadata.adc_full_scale, b.metadata.adc_full_scale));
  EXPECT_EQ(a.metadata.extra_json, b.metadata.extra_json);
}

struct TempFile {
  std::string path = temp_path("gpusonar_test");
  ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

class RoundTrip : public ::testing::TestWithParam<SampleKind> {};

TEST_P(RoundTrip, EveryFieldSurvivesBitExact) {
  TempFile f;
  const PingRecord a = make_full_record(8, 257, GetParam(), 1);
  PingRecord b = make_full_record(3, 100, GetParam(), 2);  // different shape in the same file
  b.metadata.sequence = 43;
  b.transmit.time_ns += 1'000'000'000;
  {
    io::WriterOptions opt;
    opt.config_yaml = "sonar: flank_32\n";
    io::RecordWriter w(f.path, opt);
    EXPECT_EQ(w.write(a), 0);
    EXPECT_EQ(w.write(b), 1);
  }
  io::RecordReader r(f.path);
  EXPECT_EQ(r.size(), 2);
  EXPECT_EQ(r.info().schema_name, kSchemaName);
  EXPECT_EQ(r.info().version_major, kSchemaMajor);
  EXPECT_EQ(r.info().config_yaml, "sonar: flank_32\n");
  expect_equal(a, r.read(0));
  expect_equal(b, r.read(1));
  EXPECT_TRUE(io::validate_file(f.path).ok()) << io::validate_file(f.path).to_string();
}

INSTANTIATE_TEST_SUITE_P(Kinds, RoundTrip,
                         ::testing::Values(SampleKind::RealPassband, SampleKind::ComplexBaseband,
                                           SampleKind::ComplexAnalytic));

TEST(RecordIo, HeaderOnlyReadSkipsSamples) {
  TempFile f;
  const PingRecord a = make_full_record(4, 64, SampleKind::ComplexBaseband);
  { io::RecordWriter(f.path).write(a); }
  const PingRecord h = io::RecordReader(f.path).read(0, false);
  EXPECT_TRUE(h.samples.data.empty());
  expect_equal(a, h, false);
}

TEST(RecordIo, CompressionIsLossless) {
  TempFile f;
  const PingRecord a = make_full_record(4, 4096, SampleKind::RealPassband);
  io::WriterOptions opt;
  opt.compression_level = 4;
  { io::RecordWriter(f.path, opt).write(a); }
  expect_equal(a, io::RecordReader(f.path).read(0));
}

TEST(RecordIo, Int16StorageQuantisesToTheStatedScale) {
  // 12-bit ADC counts stored as int16: values on the grid survive exactly.
  TempFile f;
  PingRecord a = make_full_record(2, 500, SampleKind::RealPassband);
  a.samples.storage = StorageType::Int16;
  a.samples.storage_scale = 1.0 / 2048.0;
  for (auto& v : a.samples.data) v = std::nearbyint(v * 2047.0F) / 2048.0F;
  { io::RecordWriter(f.path).write(a); }
  expect_equal(a, io::RecordReader(f.path).read(0));
}

TEST(RecordIo, Int32StorageHolds24BitComplexSamples) {
  TempFile f;
  PingRecord a = make_full_record(2, 300, SampleKind::ComplexBaseband);
  a.samples.storage = StorageType::Int32;
  const double scale = 1.0 / 8388608.0;  // 2^-23
  a.samples.storage_scale = scale;
  for (auto& v : a.samples.data) {
    v = static_cast<float>(std::nearbyint(static_cast<double>(v) * 8388607.0) * scale);
  }
  { io::RecordWriter(f.path).write(a); }
  const PingRecord b = io::RecordReader(f.path).read(0);
  ASSERT_EQ(a.samples.data.size(), b.samples.data.size());
  for (std::size_t i = 0; i < a.samples.data.size(); ++i) {
    // int32 -> float32 conversion rounds to 24 significant bits.
    EXPECT_NEAR(a.samples.data[i], b.samples.data[i], 1.5e-7F);
  }
}

TEST(RecordIo, TruthArraysRoundTrip) {
  TempFile f;
  {
    io::RecordWriter w(f.path);
    w.write(make_full_record(1, 8, SampleKind::RealPassband));
    w.write_truth("scatterers", {1, 2, 3, 4, 5, 6}, {2, 3});
    w.write_truth_text("scenario", "point_grid");
  }
  io::RecordReader r(f.path);
  ASSERT_EQ(r.truth_names().size(), 1U);
  EXPECT_EQ(r.truth_names()[0], "scatterers");
  std::vector<std::int64_t> shape;
  const auto v = r.read_truth("scatterers", shape);
  EXPECT_EQ(shape, (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(v, (std::vector<double>{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(r.read_truth_text("scenario"), "point_grid");
}

TEST(RecordIo, RejectsMismatchedSampleBuffer) {
  TempFile f;
  PingRecord a = make_full_record(2, 16, SampleKind::RealPassband);
  a.samples.data.pop_back();
  io::RecordWriter w(f.path);
  EXPECT_THROW(w.write(a), std::runtime_error);
}

TEST(RecordIo, RejectsForeignFilesAndFutureMajorVersions) {
  TempFile f;
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);  // keep expected HDF5 errors quiet
  {
    const hid_t file = H5Fcreate(f.path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(file);
  }
  EXPECT_THROW(io::RecordReader r(f.path), std::runtime_error);
  EXPECT_FALSE(io::validate_file(f.path).ok());

  { io::RecordWriter(f.path).write(make_full_record(1, 8, SampleKind::RealPassband)); }
  {
    const hid_t file = H5Fopen(f.path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    const hid_t attr = H5Aopen(file, "schema_version_major", H5P_DEFAULT);
    const std::int64_t future = kSchemaMajor + 1;
    H5Awrite(attr, H5T_NATIVE_INT64, &future);
    H5Aclose(attr);
    H5Fclose(file);
  }
  EXPECT_THROW(io::RecordReader r(f.path), std::runtime_error);
}

// --- Validator --------------------------------------------------------------

namespace {
bool has_error(const io::ValidationReport& rep, const std::string& field) {
  for (const auto& i : rep.issues) {
    if (i.severity == io::Severity::Error && i.field.find(field) != std::string::npos) return true;
  }
  return false;
}
}  // namespace

TEST(Validator, AcceptsAFullRecord) {
  const auto rep = io::validate(make_full_record(8, 64, SampleKind::ComplexBaseband));
  EXPECT_TRUE(rep.ok()) << rep.to_string();
  EXPECT_EQ(rep.warning_count(), 0U) << rep.to_string();
}

TEST(Validator, FlagsEachKindOfDamage) {
  const PingRecord good = make_full_record(4, 64, SampleKind::ComplexBaseband);
  {
    PingRecord r = good;
    r.samples.sample_rate = 0.0;
    EXPECT_TRUE(has_error(io::validate(r), "samples.sample_rate"));
  }
  {
    PingRecord r = good;
    r.samples.data[5] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(has_error(io::validate(r), "samples.data"));
  }
  {
    PingRecord r = good;
    r.geometry.rx.pop_back();
    EXPECT_TRUE(has_error(io::validate(r), "geometry.rx"));
  }
  {
    PingRecord r = good;
    r.geometry.rx[1].boresight = {0.0, 2.0, 0.0};
    EXPECT_TRUE(has_error(io::validate(r), "geometry.rx[1].boresight"));
  }
  {
    PingRecord r = good;
    r.calibration.gain[2] = 0.0;
    EXPECT_TRUE(has_error(io::validate(r), "calibration.gain"));
  }
  {
    PingRecord r = good;
    r.calibration.delay.clear();
    EXPECT_TRUE(has_error(io::validate(r), "calibration.delay"));
  }
  {
    PingRecord r = good;
    r.environment.sound_speed = -1.0;
    EXPECT_TRUE(has_error(io::validate(r), "environment.sound_speed"));
  }
  {
    PingRecord r = good;
    r.environment.profile_depth = {0.0, 10.0, 5.0};
    EXPECT_TRUE(has_error(io::validate(r), "environment.profile_depth"));
  }
  {
    PingRecord r = good;
    r.metadata.channel_board[0] = 7;
    EXPECT_TRUE(has_error(io::validate(r), "metadata.channel_board"));
  }
  {
    PingRecord r = good;
    r.navigation.ins[1].t = -1.0;
    EXPECT_TRUE(has_error(io::validate(r), "navigation.ins"));
  }
  {
    PingRecord r = good;
    r.transmit.waveform.duration = 0.0;
    EXPECT_TRUE(has_error(io::validate(r), "transmit.waveform.duration"));
  }
  {
    // 4 kHz of bandwidth cannot live in a 2 kHz complex baseband stream.
    PingRecord r = good;
    r.samples.sample_rate = 2000.0;
    EXPECT_TRUE(has_error(io::validate(r), "samples.sample_rate"));
  }
}

TEST(Validator, WarnsOnClipping) {
  PingRecord r = make_full_record(2, 64, SampleKind::RealPassband);
  r.metadata.adc_full_scale = 0.5;  // random data in [-1, 1] now clips
  const auto rep = io::validate(r);
  EXPECT_TRUE(rep.ok());
  EXPECT_GT(rep.warning_count(), 0U);
}
