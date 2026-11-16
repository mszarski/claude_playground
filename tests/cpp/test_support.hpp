// Shared helpers for tests.
#pragma once

#include <string>

#include "sonar/core/record.hpp"

namespace sonar::test {

/// A record with every field populated with distinct, non-default values.
PingRecord make_full_record(std::int64_t channels, std::int64_t samples, SampleKind kind,
                            std::uint32_t seed = 1);

/// Unique temporary file path (removed by the caller).
std::string temp_path(const std::string& stem);

}  // namespace sonar::test
