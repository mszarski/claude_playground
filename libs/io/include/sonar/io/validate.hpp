// Record validator: structural and physical sanity checks on a ping record.
#pragma once

#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::io {

enum class Severity { Warning, Error };

struct Issue {
  Severity severity;
  std::string field;    ///< dotted path, e.g. "samples.sample_rate"
  std::string message;
};

struct ValidationReport {
  std::vector<Issue> issues;
  bool ok() const;  ///< true when there are no errors (warnings allowed)
  std::size_t error_count() const;
  std::size_t warning_count() const;
  std::string to_string() const;
};

/// Check one record. Never throws on bad data; reports it.
ValidationReport validate(const PingRecord& record);

/// Check every record of a file plus the file-level schema attributes.
ValidationReport validate_file(const std::string& path);

}  // namespace sonar::io
