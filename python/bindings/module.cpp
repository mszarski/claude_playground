#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/io/record_file.hpp"
#include "sonar/io/validate.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

py::array samples_to_array(const Samples& s) {
  const std::vector<py::ssize_t> shape{s.n_channels, s.n_samples};
  if (s.data.size() != s.expected_size()) {
    throw std::runtime_error("samples.data size does not match channels x samples");
  }
  if (s.is_complex()) {
    py::array_t<std::complex<float>> a(shape);
    std::memcpy(a.mutable_data(), s.data.data(), s.data.size() * sizeof(float));
    return std::move(a);
  }
  py::array_t<float> a(shape);
  std::memcpy(a.mutable_data(), s.data.data(), s.data.size() * sizeof(float));
  return std::move(a);
}

void samples_from_array(Samples& s, const py::array& in) {
  if (in.ndim() != 2) throw std::invalid_argument("samples must be a 2-D array [channels, samples]");
  const bool cplx = in.dtype().kind() == 'c';
  if (cplx) {
    auto a = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>::ensure(in);
    if (!a) throw std::invalid_argument("cannot convert samples to complex64");
    if (!s.is_complex()) s.kind = SampleKind::ComplexBaseband;
    s.n_channels = a.shape(0);
    s.n_samples = a.shape(1);
    s.data.resize(s.expected_size());
    std::memcpy(s.data.data(), a.data(), s.data.size() * sizeof(float));
  } else {
    auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(in);
    if (!a) throw std::invalid_argument("cannot convert samples to float32");
    s.kind = SampleKind::RealPassband;
    s.n_channels = a.shape(0);
    s.n_samples = a.shape(1);
    s.data.resize(s.expected_size());
    std::memcpy(s.data.data(), a.data(), s.data.size() * sizeof(float));
  }
}

}  // namespace

void bind_sim(py::module_& m);
void bind_recon(py::module_& m);
#if SONAR_HAVE_MOTION
void bind_motion(py::module_& m);
#endif

PYBIND11_MODULE(_gpusonar, m) {
  m.doc() = "gpusonar native core";
  m.attr("SCHEMA_NAME") = kSchemaName;
  m.attr("SCHEMA_VERSION") = py::make_tuple(kSchemaMajor, kSchemaMinor, kSchemaPatch);
  m.def("cuda_compiled", &cuda_compiled);
  m.def("cuda_available", &cuda_available);
  m.def("cuda_device_count", &cuda_device_count);
  m.def("cuda_device_name", [](int d) { return std::string(cuda_device_name(d)); });

  m.def("rotation_ned", [](double roll, double pitch, double yaw) {
    const Mat3d r = rotation_ned(roll, pitch, yaw);
    py::array_t<double> a({3, 3});
    auto v = a.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) v(i, j) = r.m[i][j];
    return a;
  }, "Body-to-NED rotation matrix (ASASIN Eq. 2).");

  py::enum_<SampleKind>(m, "SampleKind")
      .value("RealPassband", SampleKind::RealPassband)
      .value("ComplexBaseband", SampleKind::ComplexBaseband)
      .value("ComplexAnalytic", SampleKind::ComplexAnalytic);
  py::enum_<Mode>(m, "Mode")
      .value("Unknown", Mode::Unknown)
      .value("Fls", Mode::Fls)
      .value("Sas", Mode::Sas)
      .value("Flank", Mode::Flank);
  py::enum_<WaveformType>(m, "WaveformType")
      .value("Cw", WaveformType::Cw)
      .value("Lfm", WaveformType::Lfm)
      .value("Gaussian", WaveformType::Gaussian)
      .value("Arbitrary", WaveformType::Arbitrary);
  py::enum_<TransmitKind>(m, "TransmitKind")
      .value("Point", TransmitKind::Point)
      .value("PlaneWave", TransmitKind::PlaneWave);
  py::enum_<StorageType>(m, "StorageType")
      .value("Float32", StorageType::Float32)
      .value("Int16", StorageType::Int16)
      .value("Int32", StorageType::Int32);

  py::class_<Samples>(m, "Samples")
      .def(py::init<>())
      .def_readwrite("kind", &Samples::kind)
      .def_readonly("n_channels", &Samples::n_channels)
      .def_readonly("n_samples", &Samples::n_samples)
      .def_readwrite("sample_rate", &Samples::sample_rate)
      .def_readwrite("center_frequency", &Samples::center_frequency)
      .def_readwrite("start_time", &Samples::start_time)
      .def_readwrite("storage", &Samples::storage)
      .def_readwrite("storage_scale", &Samples::storage_scale)
      .def_property("array", &samples_to_array, &samples_from_array,
                    "Samples as a [channels, samples] float32 or complex64 array (copied).")
      .def("is_complex", &Samples::is_complex);

  py::class_<Waveform>(m, "Waveform")
      .def(py::init<>())
      .def_readwrite("type", &Waveform::type)
      .def_readwrite("f_start", &Waveform::f_start)
      .def_readwrite("f_stop", &Waveform::f_stop)
      .def_readwrite("duration", &Waveform::duration)
      .def_readwrite("window_alpha", &Waveform::window_alpha)
      .def_readwrite("source_level_db", &Waveform::source_level_db)
      .def_readwrite("amplitude", &Waveform::amplitude)
      .def_readwrite("replica_sample_rate", &Waveform::replica_sample_rate)
      .def_readwrite("replica", &Waveform::replica)
      .def("center", &Waveform::center)
      .def("bandwidth", &Waveform::bandwidth);

  py::class_<Pose>(m, "Pose")
      .def(py::init<>())
      .def_readwrite("position", &Pose::position)
      .def_readwrite("rpy", &Pose::rpy)
      .def_readwrite("velocity", &Pose::velocity)
      .def_readwrite("rates", &Pose::rates);

  py::class_<Element>(m, "Element")
      .def(py::init<>())
      .def_readwrite("lever_arm", &Element::lever_arm)
      .def_readwrite("boresight", &Element::boresight)
      .def_readwrite("beamwidth_az", &Element::beamwidth_az)
      .def_readwrite("beamwidth_el", &Element::beamwidth_el);

  py::class_<Transmit>(m, "Transmit")
      .def(py::init<>())
      .def_readwrite("waveform", &Transmit::waveform)
      .def_readwrite("time_ns", &Transmit::time_ns)
      .def_readwrite("kind", &Transmit::kind)
      .def_readwrite("direction", &Transmit::direction);

  py::class_<Geometry>(m, "Geometry")
      .def(py::init<>())
      .def_readwrite("tx_pose", &Geometry::tx_pose)
      .def_readwrite("rx_pose", &Geometry::rx_pose)
      .def_readwrite("tx", &Geometry::tx)
      .def_readwrite("rx", &Geometry::rx)
      .def_readwrite("frame_origin_lla", &Geometry::frame_origin_lla);

  py::class_<InsSample>(m, "InsSample")
      .def(py::init<>())
      .def_readwrite("t", &InsSample::t)
      .def_readwrite("position", &InsSample::position)
      .def_readwrite("rpy", &InsSample::rpy)
      .def_readwrite("velocity", &InsSample::velocity)
      .def_readwrite("rates", &InsSample::rates);
  py::class_<DvlSample>(m, "DvlSample")
      .def(py::init<>())
      .def_readwrite("t", &DvlSample::t)
      .def_readwrite("velocity_body", &DvlSample::velocity_body)
      .def_readwrite("altitude", &DvlSample::altitude);
  py::class_<GpsSample>(m, "GpsSample")
      .def(py::init<>())
      .def_readwrite("t", &GpsSample::t)
      .def_readwrite("latitude", &GpsSample::latitude)
      .def_readwrite("longitude", &GpsSample::longitude)
      .def_readwrite("altitude", &GpsSample::altitude)
      .def_readwrite("horizontal_accuracy", &GpsSample::horizontal_accuracy);
  py::class_<DepthSample>(m, "DepthSample")
      .def(py::init<>())
      .def_readwrite("t", &DepthSample::t)
      .def_readwrite("depth", &DepthSample::depth);
  py::class_<Navigation>(m, "Navigation")
      .def(py::init<>())
      .def_readwrite("ins", &Navigation::ins)
      .def_readwrite("dvl", &Navigation::dvl)
      .def_readwrite("gps", &Navigation::gps)
      .def_readwrite("depth", &Navigation::depth);

  py::class_<Calibration>(m, "Calibration")
      .def(py::init<>())
      .def_readwrite("gain", &Calibration::gain)
      .def_readwrite("phase", &Calibration::phase)
      .def_readwrite("delay", &Calibration::delay)
      .def_readwrite("position_offset", &Calibration::position_offset);

  py::class_<Environment>(m, "Environment")
      .def(py::init<>())
      .def_readwrite("sound_speed", &Environment::sound_speed)
      .def_readwrite("temperature", &Environment::temperature)
      .def_readwrite("salinity", &Environment::salinity)
      .def_readwrite("water_depth", &Environment::water_depth)
      .def_readwrite("profile_depth", &Environment::profile_depth)
      .def_readwrite("profile_speed", &Environment::profile_speed);

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def_readwrite("clock_offset", &Board::clock_offset)
      .def_readwrite("clock_drift_ppm", &Board::clock_drift_ppm)
      .def_readwrite("pps_sample", &Board::pps_sample)
      .def_readwrite("reference_channel", &Board::reference_channel)
      .def_readwrite("synchronised", &Board::synchronised);

  py::class_<Metadata>(m, "Metadata")
      .def(py::init<>())
      .def_readwrite("array_id", &Metadata::array_id)
      .def_readwrite("mode", &Metadata::mode)
      .def_readwrite("sequence", &Metadata::sequence)
      .def_readwrite("sub_sequence", &Metadata::sub_sequence)
      .def_readwrite("channel_element", &Metadata::channel_element)
      .def_readwrite("channel_board", &Metadata::channel_board)
      .def_readwrite("boards", &Metadata::boards)
      .def_readwrite("hardware_gain_db", &Metadata::hardware_gain_db)
      .def_readwrite("tvg_time", &Metadata::tvg_time)
      .def_readwrite("tvg_gain_db", &Metadata::tvg_gain_db)
      .def_readwrite("adc_bits", &Metadata::adc_bits)
      .def_readwrite("adc_full_scale", &Metadata::adc_full_scale)
      .def_readwrite("extra_json", &Metadata::extra_json);

  py::class_<PingRecord>(m, "PingRecord")
      .def(py::init<>())
      .def_readwrite("samples", &PingRecord::samples)
      .def_readwrite("transmit", &PingRecord::transmit)
      .def_readwrite("geometry", &PingRecord::geometry)
      .def_readwrite("navigation", &PingRecord::navigation)
      .def_readwrite("calibration", &PingRecord::calibration)
      .def_readwrite("environment", &PingRecord::environment)
      .def_readwrite("metadata", &PingRecord::metadata)
      .def("copy", [](const PingRecord& r) { return PingRecord(r); }, "Deep copy.")
      .def("set_default_channel_tables",
           [](PingRecord& r) { set_default_channel_tables(r); },
           "Fill calibration and channel maps with neutral defaults sized to the samples.")
      .def("rx_positions", [](const PingRecord& r, double t) {
        py::array_t<double> a({static_cast<py::ssize_t>(r.geometry.rx.size()), py::ssize_t{3}});
        auto v = a.mutable_unchecked<2>();
        for (std::size_t i = 0; i < r.geometry.rx.size(); ++i) {
          Vec3d lever = r.geometry.rx[i].lever_arm;
          if (i < r.calibration.position_offset.size()) lever += r.calibration.position_offset[i];
          const Vec3d p = world_position(r.geometry.rx_pose, lever, t);
          v(i, 0) = p.x; v(i, 1) = p.y; v(i, 2) = p.z;
        }
        return a;
      }, py::arg("t") = 0.0, "World positions [channels, 3] of the receive elements at time t after transmit.")
      .def("tx_position", [](const PingRecord& r, double t) {
        return world_position(r.geometry.tx_pose, r.geometry.tx.lever_arm, t);
      }, py::arg("t") = 0.0);

  // --- io
  py::class_<io::WriterOptions>(m, "WriterOptions")
      .def(py::init<>())
      .def_readwrite("compression_level", &io::WriterOptions::compression_level)
      .def_readwrite("created_by", &io::WriterOptions::created_by)
      .def_readwrite("config_yaml", &io::WriterOptions::config_yaml);

  py::class_<io::RecordWriter>(m, "RecordWriter")
      .def(py::init<const std::string&, const io::WriterOptions&>(), py::arg("path"),
           py::arg("options") = io::WriterOptions{})
      .def("write", &io::RecordWriter::write)
      .def("write_truth", [](io::RecordWriter& w, const std::string& name,
                             py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        std::vector<std::int64_t> shape(a.shape(), a.shape() + a.ndim());
        std::vector<double> values(a.data(), a.data() + a.size());
        w.write_truth(name, values, shape);
      })
      .def("write_truth_text", &io::RecordWriter::write_truth_text)
      .def("flush", &io::RecordWriter::flush)
      .def("close", &io::RecordWriter::close)
      .def("__enter__", [](io::RecordWriter& w) -> io::RecordWriter& { return w; },
           py::return_value_policy::reference)
      .def("__exit__", [](io::RecordWriter& w, const py::object&, const py::object&,
                          const py::object&) { w.close(); });

  py::class_<io::FileInfo>(m, "FileInfo")
      .def_readonly("schema_name", &io::FileInfo::schema_name)
      .def_property_readonly("schema_version", [](const io::FileInfo& i) {
        return py::make_tuple(i.version_major, i.version_minor, i.version_patch);
      })
      .def_readonly("created_by", &io::FileInfo::created_by)
      .def_readonly("config_yaml", &io::FileInfo::config_yaml)
      .def_readonly("n_pings", &io::FileInfo::n_pings);

  py::class_<io::RecordReader>(m, "RecordReader")
      .def(py::init<const std::string&>())
      .def_property_readonly("info", &io::RecordReader::info)
      .def("__len__", &io::RecordReader::size)
      .def("read", &io::RecordReader::read, py::arg("index"), py::arg("load_samples") = true)
      .def("truth_names", &io::RecordReader::truth_names)
      .def("read_truth", [](const io::RecordReader& r, const std::string& name) {
        std::vector<std::int64_t> shape;
        const auto values = r.read_truth(name, shape);
        py::array_t<double> a(std::vector<py::ssize_t>(shape.begin(), shape.end()));
        std::memcpy(a.mutable_data(), values.data(), values.size() * sizeof(double));
        return a;
      })
      .def("read_truth_text", &io::RecordReader::read_truth_text);

  py::enum_<io::Severity>(m, "Severity")
      .value("Warning", io::Severity::Warning)
      .value("Error", io::Severity::Error);
  py::class_<io::Issue>(m, "Issue")
      .def_readonly("severity", &io::Issue::severity)
      .def_readonly("field", &io::Issue::field)
      .def_readonly("message", &io::Issue::message)
      .def("__repr__", [](const io::Issue& i) {
        return std::string(i.severity == io::Severity::Error ? "error: " : "warning: ") + i.field +
               ": " + i.message;
      });
  py::class_<io::ValidationReport>(m, "ValidationReport")
      .def_readonly("issues", &io::ValidationReport::issues)
      .def_property_readonly("ok", &io::ValidationReport::ok)
      .def_property_readonly("error_count", &io::ValidationReport::error_count)
      .def_property_readonly("warning_count", &io::ValidationReport::warning_count)
      .def("__str__", &io::ValidationReport::to_string);
  m.def("validate", &io::validate, "Check one ping record.");
  m.def("validate_file", &io::validate_file, "Check every record in a container file.");

  py::enum_<Backend>(m, "Backend").value("Host", Backend::Host).value("Cuda", Backend::Cuda);
  bind_sim(m);
  bind_recon(m);
#if SONAR_HAVE_MOTION
  bind_motion(m);
  m.attr("HAVE_MOTION") = true;
#else
  m.attr("HAVE_MOTION") = false;
#endif
}
