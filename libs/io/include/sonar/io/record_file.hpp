// HDF5 container for ping records. Layout: docs/RECORD_FORMAT.md.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::io {

struct WriterOptions {
  int compression_level = 0;  ///< gzip 0-9 on the samples dataset; 0 = off (live logging)
  std::string created_by = "gpusonar";
  std::string config_yaml;  ///< resolved processing/simulation config, stored verbatim
};

/// Appends ping records to one file. Single writer; not thread-safe.
class RecordWriter {
 public:
  explicit RecordWriter(const std::string& path, const WriterOptions& options = {});
  ~RecordWriter();
  RecordWriter(const RecordWriter&) = delete;
  RecordWriter& operator=(const RecordWriter&) = delete;

  /// Write one ping. Returns its index in the file.
  std::int64_t write(const PingRecord& record);

  /// Store a named float64 array under /truth (simulator ground truth).
  void write_truth(const std::string& name, const std::vector<double>& values,
                   const std::vector<std::int64_t>& shape);
  /// Store a string attribute under /truth.
  void write_truth_text(const std::string& name, const std::string& text);

  void flush();
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct FileInfo {
  std::string schema_name;
  int version_major = 0, version_minor = 0, version_patch = 0;
  std::string created_by;
  std::string config_yaml;
  std::int64_t n_pings = 0;
};

/// Reads a record file. Errors are exceptions:
///   - the constructor throws std::runtime_error when the file cannot be opened as HDF5, when
///     its schema_name is not this schema's, or when its schema major version differs from the
///     reader's (a different minor or patch version is accepted);
///   - read() throws std::out_of_range for an index outside [0, size()), and
///     std::runtime_error for a ping whose contents are malformed (missing or inconsistent
///     datasets, an unknown storage type);
///   - read_truth() throws std::runtime_error when the file has no /truth group or no array of
///     that name; truth_names() returns an empty list and read_truth_text() an empty string
///     in those cases instead.
/// Any other HDF5 failure surfaces as std::runtime_error("HDF5: failed to ...").
class RecordReader {
 public:
  explicit RecordReader(const std::string& path);
  ~RecordReader();
  RecordReader(const RecordReader&) = delete;
  RecordReader& operator=(const RecordReader&) = delete;

  const FileInfo& info() const;
  std::int64_t size() const;

  /// Read ping `index`. With `load_samples == false` only the header fields are read.
  PingRecord read(std::int64_t index, bool load_samples = true) const;

  std::vector<std::string> truth_names() const;
  /// Read a /truth array; `shape` receives its dimensions.
  std::vector<double> read_truth(const std::string& name, std::vector<std::int64_t>& shape) const;
  std::string read_truth_text(const std::string& name) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonar::io
