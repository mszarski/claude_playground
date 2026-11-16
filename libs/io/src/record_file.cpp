#include "sonar/io/record_file.hpp"

#include <hdf5.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace sonar::io {
namespace {

// --- Minimal RAII and helpers over the HDF5 C API ---------------------------

enum class Kind { File, Group, Dataset, Space, Type, Attr, Plist };

class Hid {
 public:
  Hid() = default;
  Hid(hid_t id, Kind kind, const std::string& what) : id_(id), kind_(kind) {
    if (id_ < 0) throw std::runtime_error("HDF5: failed to " + what);
  }
  Hid(Hid&& o) noexcept : id_(std::exchange(o.id_, -1)), kind_(o.kind_) {}
  Hid& operator=(Hid&& o) noexcept {
    if (this != &o) {
      reset();
      id_ = std::exchange(o.id_, -1);
      kind_ = o.kind_;
    }
    return *this;
  }
  Hid(const Hid&) = delete;
  Hid& operator=(const Hid&) = delete;
  ~Hid() { reset(); }
  void reset() {
    if (id_ < 0) return;
    switch (kind_) {
      case Kind::File: H5Fclose(id_); break;
      case Kind::Group: H5Gclose(id_); break;
      case Kind::Dataset: H5Dclose(id_); break;
      case Kind::Space: H5Sclose(id_); break;
      case Kind::Type: H5Tclose(id_); break;
      case Kind::Attr: H5Aclose(id_); break;
      case Kind::Plist: H5Pclose(id_); break;
    }
    id_ = -1;
  }
  operator hid_t() const { return id_; }  // NOLINT(google-explicit-constructor)
  bool valid() const { return id_ >= 0; }

 private:
  hid_t id_ = -1;
  Kind kind_ = Kind::File;
};

void check(herr_t status, const std::string& what) {
  if (status < 0) throw std::runtime_error("HDF5: failed to " + what);
}

Hid make_group(hid_t parent, const std::string& name) {
  return {H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), Kind::Group,
          "create group " + name};
}
Hid open_group(hid_t parent, const std::string& name) {
  return {H5Gopen2(parent, name.c_str(), H5P_DEFAULT), Kind::Group, "open group " + name};
}
bool exists(hid_t parent, const std::string& name) {
  return H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0;
}

template <typename T>
hid_t native_type();
template <>
hid_t native_type<double>() { return H5T_NATIVE_DOUBLE; }
template <>
hid_t native_type<std::int64_t>() { return H5T_NATIVE_INT64; }
template <>
hid_t native_type<std::int32_t>() { return H5T_NATIVE_INT32; }

template <typename T>
void write_attr(hid_t obj, const std::string& name, T value) {
  Hid space(H5Screate(H5S_SCALAR), Kind::Space, "create scalar space");
  Hid attr(H5Acreate2(obj, name.c_str(), native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT),
           Kind::Attr, "create attribute " + name);
  check(H5Awrite(attr, native_type<T>(), &value), "write attribute " + name);
}

void write_attr(hid_t obj, const std::string& name, const std::string& value) {
  Hid type(H5Tcopy(H5T_C_S1), Kind::Type, "copy string type");
  check(H5Tset_size(type, H5T_VARIABLE), "set string size");
  check(H5Tset_cset(type, H5T_CSET_UTF8), "set string charset");
  Hid space(H5Screate(H5S_SCALAR), Kind::Space, "create scalar space");
  Hid attr(H5Acreate2(obj, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), Kind::Attr,
           "create attribute " + name);
  const char* ptr = value.c_str();
  check(H5Awrite(attr, type, &ptr), "write attribute " + name);
}

template <typename T>
T read_attr(hid_t obj, const std::string& name) {
  Hid attr(H5Aopen(obj, name.c_str(), H5P_DEFAULT), Kind::Attr, "open attribute " + name);
  T value{};
  check(H5Aread(attr, native_type<T>(), &value), "read attribute " + name);
  return value;
}

template <typename T>
T read_attr_or(hid_t obj, const std::string& name, T fallback) {
  if (H5Aexists(obj, name.c_str()) <= 0) return fallback;
  return read_attr<T>(obj, name);
}

std::string read_attr_string(hid_t obj, const std::string& name, const std::string& fallback = "") {
  if (H5Aexists(obj, name.c_str()) <= 0) return fallback;
  Hid attr(H5Aopen(obj, name.c_str(), H5P_DEFAULT), Kind::Attr, "open attribute " + name);
  Hid ftype(H5Aget_type(attr), Kind::Type, "get attribute type");
  std::string out;
  if (H5Tis_variable_str(ftype) > 0) {
    Hid type(H5Tcopy(H5T_C_S1), Kind::Type, "copy string type");
    check(H5Tset_size(type, H5T_VARIABLE), "set string size");
    check(H5Tset_cset(type, H5Tget_cset(ftype)), "set string charset");
    char* ptr = nullptr;
    check(H5Aread(attr, type, &ptr), "read attribute " + name);
    if (ptr != nullptr) {
      out = ptr;
      H5free_memory(ptr);
    }
  } else {
    const std::size_t n = H5Tget_size(ftype);
    std::vector<char> buf(n + 1, '\0');
    Hid type(H5Tcopy(H5T_C_S1), Kind::Type, "copy string type");
    check(H5Tset_size(type, n), "set string size");
    check(H5Aread(attr, type, buf.data()), "read attribute " + name);
    out = buf.data();
  }
  return out;
}

template <typename T>
void write_array(hid_t parent, const std::string& name, const T* data,
                 const std::vector<hsize_t>& dims) {
  hsize_t total = 1;
  for (auto d : dims) total *= d;
  if (total == 0) return;  // absent dataset == empty
  Hid space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), Kind::Space,
            "create dataspace for " + name);
  Hid ds(H5Dcreate2(parent, name.c_str(), native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT,
                    H5P_DEFAULT),
         Kind::Dataset, "create dataset " + name);
  check(H5Dwrite(ds, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
        "write dataset " + name);
}

template <typename T>
void write_array(hid_t parent, const std::string& name, const std::vector<T>& data) {
  write_array(parent, name, data.data(), {static_cast<hsize_t>(data.size())});
}

template <typename T>
std::vector<T> read_array(hid_t parent, const std::string& name, std::vector<hsize_t>* dims_out = nullptr) {
  std::vector<T> out;
  if (dims_out != nullptr) dims_out->clear();
  if (!exists(parent, name)) return out;
  Hid ds(H5Dopen2(parent, name.c_str(), H5P_DEFAULT), Kind::Dataset, "open dataset " + name);
  Hid space(H5Dget_space(ds), Kind::Space, "get dataspace of " + name);
  const int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank > 0 ? rank : 0));
  if (rank > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
  hsize_t total = 1;
  for (auto d : dims) total *= d;
  out.resize(total);
  if (total > 0) {
    check(H5Dread(ds, native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()),
          "read dataset " + name);
  }
  if (dims_out != nullptr) *dims_out = dims;
  return out;
}

void put3(std::vector<double>& v, const Vec3d& a) {
  v.push_back(a.x);
  v.push_back(a.y);
  v.push_back(a.z);
}
Vec3d get3(const std::vector<double>& v, std::size_t offset) {
  return {v.at(offset), v.at(offset + 1), v.at(offset + 2)};
}

void write_pose(hid_t parent, const std::string& name, const Pose& p) {
  std::vector<double> v;
  put3(v, p.position);
  put3(v, p.rpy);
  put3(v, p.velocity);
  put3(v, p.rates);
  write_array(parent, name, v.data(), {4, 3});
}
Pose read_pose(hid_t parent, const std::string& name) {
  const auto v = read_array<double>(parent, name);
  if (v.size() != 12) throw std::runtime_error("record: " + name + " must be 4x3");
  return {get3(v, 0), get3(v, 3), get3(v, 6), get3(v, 9)};
}

std::string ping_name(std::int64_t index) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%08lld", static_cast<long long>(index));
  return buf;
}

hid_t storage_base_type(StorageType s) {
  switch (s) {
    case StorageType::Float32: return H5T_NATIVE_FLOAT;
    case StorageType::Int16: return H5T_NATIVE_INT16;
    case StorageType::Int32: return H5T_NATIVE_INT32;
  }
  throw std::runtime_error("record: unknown storage type");
}
std::size_t storage_size(StorageType s) {
  switch (s) {
    case StorageType::Float32: return 4;
    case StorageType::Int16: return 2;
    case StorageType::Int32: return 4;
  }
  return 4;
}

/// Element type for the samples dataset: scalar for real data, {r, i} compound for
/// complex (the layout h5py reads as a complex or structured array).
Hid sample_type(StorageType s, bool is_complex) {
  const hid_t base = storage_base_type(s);
  if (!is_complex) return {H5Tcopy(base), Kind::Type, "copy sample type"};
  const std::size_t n = storage_size(s);
  Hid type(H5Tcreate(H5T_COMPOUND, 2 * n), Kind::Type, "create complex type");
  check(H5Tinsert(type, "r", 0, base), "insert r");
  check(H5Tinsert(type, "i", n, base), "insert i");
  return type;
}

template <typename I>
std::vector<I> quantise(const std::vector<float>& data, double scale) {
  std::vector<I> out(data.size());
  const double inv = 1.0 / scale;
  const double lo = static_cast<double>(std::numeric_limits<I>::min());
  const double hi = static_cast<double>(std::numeric_limits<I>::max());
  for (std::size_t i = 0; i < data.size(); ++i) {
    double v = std::nearbyint(static_cast<double>(data[i]) * inv);
    v = v < lo ? lo : (v > hi ? hi : v);
    out[i] = static_cast<I>(v);
  }
  return out;
}

}  // namespace

// --- Writer -----------------------------------------------------------------

struct RecordWriter::Impl {
  Hid file;
  Hid pings;
  Hid truth;
  WriterOptions options;
  std::int64_t count = 0;
};

RecordWriter::RecordWriter(const std::string& path, const WriterOptions& options)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = options;
  impl_->file = Hid(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), Kind::File,
                    "create file " + path);
  write_attr(impl_->file, "schema_name", std::string(kSchemaName));
  write_attr<std::int64_t>(impl_->file, "schema_version_major", kSchemaMajor);
  write_attr<std::int64_t>(impl_->file, "schema_version_minor", kSchemaMinor);
  write_attr<std::int64_t>(impl_->file, "schema_version_patch", kSchemaPatch);
  write_attr(impl_->file, "created_by", options.created_by);
  write_attr(impl_->file, "config_yaml", options.config_yaml);
  impl_->pings = make_group(impl_->file, "pings");
}

RecordWriter::~RecordWriter() {
  try {
    close();
  } catch (...) {  // NOLINT(bugprone-empty-catch) destructors must not throw
  }
}

std::int64_t RecordWriter::write(const PingRecord& rec) {
  if (!impl_->file.valid()) throw std::runtime_error("record writer is closed");
  const Samples& s = rec.samples;
  if (s.data.size() != s.expected_size()) {
    throw std::runtime_error("record: samples.data size does not match channels x samples");
  }
  const std::int64_t index = impl_->count;
  Hid g = make_group(impl_->pings, ping_name(index));

  // samples
  {
    const std::vector<hsize_t> dims{static_cast<hsize_t>(s.n_channels),
                                    static_cast<hsize_t>(s.n_samples)};
    if (s.n_channels > 0 && s.n_samples > 0) {
      Hid space(H5Screate_simple(2, dims.data(), nullptr), Kind::Space, "create samples space");
      Hid type = sample_type(s.storage, s.is_complex());
      Hid plist(H5Pcreate(H5P_DATASET_CREATE), Kind::Plist, "create dataset plist");
      if (impl_->options.compression_level > 0) {
        const hsize_t chunk[2] = {1, dims[1]};
        check(H5Pset_chunk(plist, 2, chunk), "set chunking");
        check(H5Pset_shuffle(plist), "set shuffle");
        check(H5Pset_deflate(plist, static_cast<unsigned>(impl_->options.compression_level)),
              "set deflate");
      }
      Hid ds(H5Dcreate2(g, "samples", type, space, H5P_DEFAULT, plist, H5P_DEFAULT),
             Kind::Dataset, "create samples dataset");
      switch (s.storage) {
        case StorageType::Float32:
          check(H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, s.data.data()), "write samples");
          break;
        case StorageType::Int16: {
          const auto q = quantise<std::int16_t>(s.data, s.storage_scale);
          check(H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, q.data()), "write samples");
          break;
        }
        case StorageType::Int32: {
          const auto q = quantise<std::int32_t>(s.data, s.storage_scale);
          check(H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, q.data()), "write samples");
          break;
        }
      }
    }
    write_attr<std::int64_t>(g, "n_channels", s.n_channels);
    write_attr<std::int64_t>(g, "n_samples", s.n_samples);
    write_attr<std::int64_t>(g, "sample_kind", static_cast<std::int64_t>(s.kind));
    write_attr(g, "sample_rate", s.sample_rate);
    write_attr(g, "center_frequency", s.center_frequency);
    write_attr(g, "start_time", s.start_time);
    write_attr<std::int64_t>(g, "storage_type", static_cast<std::int64_t>(s.storage));
    write_attr(g, "storage_scale", s.storage_scale);
  }

  // transmit
  {
    Hid t = make_group(g, "transmit");
    const Waveform& w = rec.transmit.waveform;
    write_attr<std::int64_t>(t, "time_ns", rec.transmit.time_ns);
    write_attr<std::int64_t>(t, "kind", static_cast<std::int64_t>(rec.transmit.kind));
    const double dir[3] = {rec.transmit.direction.x, rec.transmit.direction.y,
                           rec.transmit.direction.z};
    write_array(t, "direction", dir, {3});
    write_attr<std::int64_t>(t, "waveform_type", static_cast<std::int64_t>(w.type));
    write_attr(t, "f_start", w.f_start);
    write_attr(t, "f_stop", w.f_stop);
    write_attr(t, "duration", w.duration);
    write_attr(t, "window_alpha", w.window_alpha);
    write_attr(t, "source_level_db", w.source_level_db);
    write_attr(t, "amplitude", w.amplitude);
    write_attr(t, "replica_sample_rate", w.replica_sample_rate);
    if (!w.replica.empty()) {
      const std::vector<double> rep(w.replica.begin(), w.replica.end());
      write_array(t, "replica", rep);
    }
  }

  // geometry
  {
    Hid geo = make_group(g, "geometry");
    const Geometry& q = rec.geometry;
    write_pose(geo, "tx_pose", q.tx_pose);
    write_pose(geo, "rx_pose", q.rx_pose);
    std::vector<double> v;
    put3(v, q.tx.lever_arm);
    write_array(geo, "tx_lever_arm", v.data(), {3});
    v.clear();
    put3(v, q.tx.boresight);
    write_array(geo, "tx_boresight", v.data(), {3});
    const double txbw[2] = {q.tx.beamwidth_az, q.tx.beamwidth_el};
    write_array(geo, "tx_beamwidth", txbw, {2});
    const auto n = static_cast<hsize_t>(q.rx.size());
    std::vector<double> lever, bore, bw;
    for (const auto& e : q.rx) {
      put3(lever, e.lever_arm);
      put3(bore, e.boresight);
      bw.push_back(e.beamwidth_az);
      bw.push_back(e.beamwidth_el);
    }
    write_array(geo, "rx_lever_arm", lever.data(), {n, 3});
    write_array(geo, "rx_boresight", bore.data(), {n, 3});
    write_array(geo, "rx_beamwidth", bw.data(), {n, 2});
    v.clear();
    put3(v, q.frame_origin_lla);
    write_array(geo, "frame_origin_lla", v.data(), {3});
  }

  // navigation
  {
    Hid nav = make_group(g, "navigation");
    std::vector<double> v;
    for (const auto& a : rec.navigation.ins) {
      v.push_back(a.t);
      put3(v, a.position);
      put3(v, a.rpy);
      put3(v, a.velocity);
      put3(v, a.rates);
    }
    write_array(nav, "ins", v.data(), {static_cast<hsize_t>(rec.navigation.ins.size()), 13});
    v.clear();
    for (const auto& a : rec.navigation.dvl) {
      v.push_back(a.t);
      put3(v, a.velocity_body);
      v.push_back(a.altitude);
    }
    write_array(nav, "dvl", v.data(), {static_cast<hsize_t>(rec.navigation.dvl.size()), 5});
    v.clear();
    for (const auto& a : rec.navigation.gps) {
      v.insert(v.end(), {a.t, a.latitude, a.longitude, a.altitude, a.horizontal_accuracy});
    }
    write_array(nav, "gps", v.data(), {static_cast<hsize_t>(rec.navigation.gps.size()), 5});
    v.clear();
    for (const auto& a : rec.navigation.depth) v.insert(v.end(), {a.t, a.depth});
    write_array(nav, "depth", v.data(), {static_cast<hsize_t>(rec.navigation.depth.size()), 2});
  }

  // calibration
  {
    Hid cal = make_group(g, "calibration");
    write_array(cal, "gain", rec.calibration.gain);
    write_array(cal, "phase", rec.calibration.phase);
    write_array(cal, "delay", rec.calibration.delay);
    std::vector<double> v;
    for (const auto& p : rec.calibration.position_offset) put3(v, p);
    write_array(cal, "position_offset", v.data(),
                {static_cast<hsize_t>(rec.calibration.position_offset.size()), 3});
  }

  // environment
  {
    Hid env = make_group(g, "environment");
    write_attr(env, "sound_speed", rec.environment.sound_speed);
    write_attr(env, "temperature", rec.environment.temperature);
    write_attr(env, "salinity", rec.environment.salinity);
    write_attr(env, "water_depth", rec.environment.water_depth);
    write_array(env, "profile_depth", rec.environment.profile_depth);
    write_array(env, "profile_speed", rec.environment.profile_speed);
  }

  // metadata
  {
    Hid meta = make_group(g, "metadata");
    const Metadata& m = rec.metadata;
    write_attr(meta, "array_id", m.array_id);
    write_attr<std::int64_t>(meta, "mode", static_cast<std::int64_t>(m.mode));
    write_attr<std::int64_t>(meta, "sequence", m.sequence);
    write_attr<std::int64_t>(meta, "sub_sequence", m.sub_sequence);
    write_attr(meta, "hardware_gain_db", m.hardware_gain_db);
    write_attr<std::int64_t>(meta, "adc_bits", m.adc_bits);
    write_attr(meta, "adc_full_scale", m.adc_full_scale);
    write_attr(meta, "extra_json", m.extra_json);
    write_array(meta, "channel_element", m.channel_element);
    write_array(meta, "channel_board", m.channel_board);
    write_array(meta, "tvg_time", m.tvg_time);
    write_array(meta, "tvg_gain_db", m.tvg_gain_db);
    std::vector<double> v;
    for (const auto& b : m.boards) {
      v.insert(v.end(), {b.clock_offset, b.clock_drift_ppm, b.pps_sample,
                         static_cast<double>(b.reference_channel),
                         static_cast<double>(b.synchronised)});
    }
    write_array(meta, "boards", v.data(), {static_cast<hsize_t>(m.boards.size()), 5});
  }

  ++impl_->count;
  return index;
}

void RecordWriter::write_truth(const std::string& name, const std::vector<double>& values,
                               const std::vector<std::int64_t>& shape) {
  if (!impl_->file.valid()) throw std::runtime_error("record writer is closed");
  if (!impl_->truth.valid()) impl_->truth = make_group(impl_->file, "truth");
  std::vector<hsize_t> dims(shape.begin(), shape.end());
  hsize_t total = 1;
  for (auto d : dims) total *= d;
  if (total != values.size()) throw std::runtime_error("truth: shape does not match data size");
  write_array(impl_->truth, name, values.data(), dims);
}

void RecordWriter::write_truth_text(const std::string& name, const std::string& text) {
  if (!impl_->file.valid()) throw std::runtime_error("record writer is closed");
  if (!impl_->truth.valid()) impl_->truth = make_group(impl_->file, "truth");
  write_attr(impl_->truth, name, text);
}

void RecordWriter::flush() {
  if (impl_->file.valid()) check(H5Fflush(impl_->file, H5F_SCOPE_GLOBAL), "flush file");
}

void RecordWriter::close() {
  impl_->truth.reset();
  impl_->pings.reset();
  impl_->file.reset();
}

// --- Reader -----------------------------------------------------------------

struct RecordReader::Impl {
  Hid file;
  Hid pings;
  FileInfo info;
};

RecordReader::RecordReader(const std::string& path) : impl_(std::make_unique<Impl>()) {
  impl_->file = Hid(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), Kind::File,
                    "open file " + path);
  FileInfo& info = impl_->info;
  info.schema_name = read_attr_string(impl_->file, "schema_name");
  if (info.schema_name != kSchemaName) {
    throw std::runtime_error("record file: schema_name is '" + info.schema_name +
                             "', expected '" + kSchemaName + "'");
  }
  info.version_major = static_cast<int>(read_attr<std::int64_t>(impl_->file, "schema_version_major"));
  info.version_minor = static_cast<int>(read_attr<std::int64_t>(impl_->file, "schema_version_minor"));
  info.version_patch = static_cast<int>(read_attr<std::int64_t>(impl_->file, "schema_version_patch"));
  if (info.version_major != kSchemaMajor) {
    throw std::runtime_error("record file: schema major version " +
                             std::to_string(info.version_major) + " is not supported (reader is " +
                             std::to_string(kSchemaMajor) + ")");
  }
  info.created_by = read_attr_string(impl_->file, "created_by");
  info.config_yaml = read_attr_string(impl_->file, "config_yaml");
  impl_->pings = open_group(impl_->file, "pings");
  H5G_info_t ginfo;
  check(H5Gget_info(impl_->pings, &ginfo), "get pings group info");
  info.n_pings = static_cast<std::int64_t>(ginfo.nlinks);
}

RecordReader::~RecordReader() = default;

const FileInfo& RecordReader::info() const { return impl_->info; }
std::int64_t RecordReader::size() const { return impl_->info.n_pings; }

PingRecord RecordReader::read(std::int64_t index, bool load_samples) const {
  if (index < 0 || index >= impl_->info.n_pings) {
    throw std::out_of_range("record file: ping index out of range");
  }
  Hid g = open_group(impl_->pings, ping_name(index));
  PingRecord rec;

  {
    Samples& s = rec.samples;
    s.n_channels = read_attr<std::int64_t>(g, "n_channels");
    s.n_samples = read_attr<std::int64_t>(g, "n_samples");
    s.kind = static_cast<SampleKind>(read_attr<std::int64_t>(g, "sample_kind"));
    s.sample_rate = read_attr<double>(g, "sample_rate");
    s.center_frequency = read_attr<double>(g, "center_frequency");
    s.start_time = read_attr<double>(g, "start_time");
    s.storage = static_cast<StorageType>(read_attr<std::int64_t>(g, "storage_type"));
    s.storage_scale = read_attr<double>(g, "storage_scale");
    if (load_samples && exists(g, "samples")) {
      Hid ds(H5Dopen2(g, "samples", H5P_DEFAULT), Kind::Dataset, "open samples");
      Hid space(H5Dget_space(ds), Kind::Space, "get samples space");
      hsize_t dims[2] = {0, 0};
      if (H5Sget_simple_extent_ndims(space) != 2) {
        throw std::runtime_error("record: samples must be 2-D");
      }
      H5Sget_simple_extent_dims(space, dims, nullptr);
      if (static_cast<std::int64_t>(dims[0]) != s.n_channels ||
          static_cast<std::int64_t>(dims[1]) != s.n_samples) {
        throw std::runtime_error("record: samples shape disagrees with n_channels/n_samples");
      }
      s.data.resize(s.expected_size());
      // Always read as float32 (HDF5 converts integer storage), then rescale.
      Hid mem = sample_type(StorageType::Float32, s.is_complex());
      check(H5Dread(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, s.data.data()), "read samples");
      if (s.storage != StorageType::Float32 && s.storage_scale != 1.0) {
        const auto scale = static_cast<float>(s.storage_scale);
        for (auto& v : s.data) v *= scale;
      }
    }
  }

  {
    Hid t = open_group(g, "transmit");
    Waveform& w = rec.transmit.waveform;
    rec.transmit.time_ns = read_attr<std::int64_t>(t, "time_ns");
    rec.transmit.kind = static_cast<TransmitKind>(read_attr<std::int64_t>(t, "kind"));
    const auto dir = read_array<double>(t, "direction");
    if (dir.size() == 3) rec.transmit.direction = get3(dir, 0);
    w.type = static_cast<WaveformType>(read_attr<std::int64_t>(t, "waveform_type"));
    w.f_start = read_attr<double>(t, "f_start");
    w.f_stop = read_attr<double>(t, "f_stop");
    w.duration = read_attr<double>(t, "duration");
    w.window_alpha = read_attr<double>(t, "window_alpha");
    w.source_level_db = read_attr<double>(t, "source_level_db");
    w.amplitude = read_attr<double>(t, "amplitude");
    w.replica_sample_rate = read_attr<double>(t, "replica_sample_rate");
    const auto rep = read_array<double>(t, "replica");
    w.replica.assign(rep.begin(), rep.end());
  }

  {
    Hid geo = open_group(g, "geometry");
    Geometry& q = rec.geometry;
    q.tx_pose = read_pose(geo, "tx_pose");
    q.rx_pose = read_pose(geo, "rx_pose");
    q.tx.lever_arm = get3(read_array<double>(geo, "tx_lever_arm"), 0);
    q.tx.boresight = get3(read_array<double>(geo, "tx_boresight"), 0);
    const auto txbw = read_array<double>(geo, "tx_beamwidth");
    q.tx.beamwidth_az = txbw.at(0);
    q.tx.beamwidth_el = txbw.at(1);
    const auto lever = read_array<double>(geo, "rx_lever_arm");
    const auto bore = read_array<double>(geo, "rx_boresight");
    const auto bw = read_array<double>(geo, "rx_beamwidth");
    const std::size_t n = lever.size() / 3;
    if (bore.size() != 3 * n || bw.size() != 2 * n) {
      throw std::runtime_error("record: receive element tables have inconsistent sizes");
    }
    q.rx.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      q.rx[i].lever_arm = get3(lever, 3 * i);
      q.rx[i].boresight = get3(bore, 3 * i);
      q.rx[i].beamwidth_az = bw[2 * i];
      q.rx[i].beamwidth_el = bw[2 * i + 1];
    }
    const auto lla = read_array<double>(geo, "frame_origin_lla");
    if (lla.size() == 3) q.frame_origin_lla = get3(lla, 0);
  }

  {
    Hid nav = open_group(g, "navigation");
    auto v = read_array<double>(nav, "ins");
    for (std::size_t i = 0; i + 13 <= v.size(); i += 13) {
      rec.navigation.ins.push_back(
          {v[i], get3(v, i + 1), get3(v, i + 4), get3(v, i + 7), get3(v, i + 10)});
    }
    v = read_array<double>(nav, "dvl");
    for (std::size_t i = 0; i + 5 <= v.size(); i += 5) {
      rec.navigation.dvl.push_back({v[i], get3(v, i + 1), v[i + 4]});
    }
    v = read_array<double>(nav, "gps");
    for (std::size_t i = 0; i + 5 <= v.size(); i += 5) {
      rec.navigation.gps.push_back({v[i], v[i + 1], v[i + 2], v[i + 3], v[i + 4]});
    }
    v = read_array<double>(nav, "depth");
    for (std::size_t i = 0; i + 2 <= v.size(); i += 2) {
      rec.navigation.depth.push_back({v[i], v[i + 1]});
    }
  }

  {
    Hid cal = open_group(g, "calibration");
    rec.calibration.gain = read_array<double>(cal, "gain");
    rec.calibration.phase = read_array<double>(cal, "phase");
    rec.calibration.delay = read_array<double>(cal, "delay");
    const auto v = read_array<double>(cal, "position_offset");
    for (std::size_t i = 0; i + 3 <= v.size(); i += 3) {
      rec.calibration.position_offset.push_back(get3(v, i));
    }
  }

  {
    Hid env = open_group(g, "environment");
    rec.environment.sound_speed = read_attr<double>(env, "sound_speed");
    rec.environment.temperature = read_attr<double>(env, "temperature");
    rec.environment.salinity = read_attr<double>(env, "salinity");
    rec.environment.water_depth = read_attr<double>(env, "water_depth");
    rec.environment.profile_depth = read_array<double>(env, "profile_depth");
    rec.environment.profile_speed = read_array<double>(env, "profile_speed");
  }

  {
    Hid meta = open_group(g, "metadata");
    Metadata& m = rec.metadata;
    m.array_id = read_attr_string(meta, "array_id");
    m.mode = static_cast<Mode>(read_attr<std::int64_t>(meta, "mode"));
    m.sequence = read_attr<std::int64_t>(meta, "sequence");
    m.sub_sequence = static_cast<std::int32_t>(read_attr<std::int64_t>(meta, "sub_sequence"));
    m.hardware_gain_db = read_attr<double>(meta, "hardware_gain_db");
    m.adc_bits = static_cast<std::int32_t>(read_attr<std::int64_t>(meta, "adc_bits"));
    m.adc_full_scale = read_attr<double>(meta, "adc_full_scale");
    m.extra_json = read_attr_string(meta, "extra_json", "{}");
    m.channel_element = read_array<std::int32_t>(meta, "channel_element");
    m.channel_board = read_array<std::int32_t>(meta, "channel_board");
    m.tvg_time = read_array<double>(meta, "tvg_time");
    m.tvg_gain_db = read_array<double>(meta, "tvg_gain_db");
    const auto v = read_array<double>(meta, "boards");
    for (std::size_t i = 0; i + 5 <= v.size(); i += 5) {
      Board b;
      b.clock_offset = v[i];
      b.clock_drift_ppm = v[i + 1];
      b.pps_sample = v[i + 2];
      b.reference_channel = static_cast<std::int32_t>(v[i + 3]);
      b.synchronised = static_cast<std::int32_t>(v[i + 4]);
      m.boards.push_back(b);
    }
  }
  return rec;
}

std::vector<std::string> RecordReader::truth_names() const {
  std::vector<std::string> names;
  if (!exists(impl_->file, "truth")) return names;
  Hid truth = open_group(impl_->file, "truth");
  H5G_info_t ginfo;
  check(H5Gget_info(truth, &ginfo), "get truth group info");
  for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
    const ssize_t len =
        H5Lget_name_by_idx(truth, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
    if (len < 0) continue;
    std::string name(static_cast<std::size_t>(len) + 1, '\0');
    H5Lget_name_by_idx(truth, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size(),
                       H5P_DEFAULT);
    name.resize(static_cast<std::size_t>(len));
    names.push_back(name);
  }
  return names;
}

std::vector<double> RecordReader::read_truth(const std::string& name,
                                             std::vector<std::int64_t>& shape) const {
  if (!exists(impl_->file, "truth")) throw std::runtime_error("record file has no /truth group");
  Hid truth = open_group(impl_->file, "truth");
  if (!exists(truth, name)) throw std::runtime_error("truth: no array named " + name);
  std::vector<hsize_t> dims;
  auto out = read_array<double>(truth, name, &dims);
  shape.assign(dims.begin(), dims.end());
  return out;
}

std::string RecordReader::read_truth_text(const std::string& name) const {
  if (!exists(impl_->file, "truth")) return {};
  Hid truth = open_group(impl_->file, "truth");
  return read_attr_string(truth, name);
}

}  // namespace sonar::io
