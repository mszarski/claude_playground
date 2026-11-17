// The chain's settings as key=value text: what build/bin/sonar_live, the Holoscan application
// and `tools/sonar_live.py --export-template` share. The keys are listed at the top of
// apps/sonar_live.cpp and in docs/CONFIGURATION.md.
#pragma once

#include <string>

#include "sonar/live/chain.hpp"

namespace sonar::live {

/// Set one option on the active chain and/or the passive options. Throws on an unknown key.
void set_option(ChainOptions& chain, PassiveOptions& passive, const std::string& key, const std::string& value);
/// The same from "key=value".
void set_pair(ChainOptions& chain, PassiveOptions& passive, const std::string& pair);
/// Every line of a file as key=value; # starts a comment.
void load_options(ChainOptions& chain, PassiveOptions& passive, const std::string& path);

}  // namespace sonar::live
