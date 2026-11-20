// The chain's settings as key=value text: what build/bin/sonar_live, the Holoscan application
// and `tools/sonar_live.py --export-template` share. The keys are listed at the top of
// apps/sonar_live.cpp and in docs/CONFIGURATION.md. A value must be what its key expects (option_value.hpp):
// a number, a whole number, 0 / 1 / true / false, a comma-separated list or one of a key's words; anything
// else (`{}`, `[0.0, 0.017]` with brackets is a list and fine, `None`) throws std::invalid_argument naming the key.
#pragma once

#include <istream>
#include <string>

#include "sonar/live/chain.hpp"

namespace sonar::live {

/// Set one option on the active chain and/or the passive options. Throws on an unknown key.
void set_option(ChainOptions& chain, PassiveOptions& passive, const std::string& key, const std::string& value);
/// The same from "key=value".
void set_pair(ChainOptions& chain, PassiveOptions& passive, const std::string& pair);
/// Every line of a file as key=value; # starts a comment.
void load_options(ChainOptions& chain, PassiveOptions& passive, const std::string& path);
/// The same from a stream (text already in memory).
void read_options(ChainOptions& chain, PassiveOptions& passive, std::istream& in);
/// `backend`: "host", "cuda" or "auto" (CUDA when the build has it and a device is present, else the host: the
/// meaning gpusonar.live gives it). Anything else throws. A chain given no backend runs on the host.
Backend parse_backend(const std::string& value);
/// Every setting of the active chain that a key=value option sets, as one JSON object keyed by the option's name
/// and in its units (degrees where the name says so; derived sizes not yet resolved, and other NaN, are null).
/// Used to check that the two ways of configuring the chain (the Python binding's dictionary and key=value text)
/// give the same chain.
std::string options_json(const ChainOptions& chain);

}  // namespace sonar::live
