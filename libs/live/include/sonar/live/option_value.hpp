// How the text of one key=value option is read (options.cpp, and the stages' own setters: tonal,
// multipath_depth, coverage). Every reader refuses text that is not what the key expects, naming the
// key, instead of reading it as 0 the way atof does: `decimate={}` or `elevation_profile_ensemble=True`
// written by a tool that passed Python text on would otherwise switch a stage off without a word.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sonar::live {

/// A number (as strtod reads it, the whole text; blanks either side allowed). Throws std::invalid_argument.
double option_number(const std::string& key, const std::string& value);
/// A whole number (an integer written as such or as a float with no fraction: 3, 3.0, 1e3).
std::int64_t option_integer(const std::string& key, const std::string& value);
/// A switch: a number (0 off, any other on) or true / false / True / False.
bool option_flag(const std::string& key, const std::string& value);
/// Numbers separated by commas (blanks and one pair of enclosing brackets allowed); empty text: an empty list.
std::vector<double> option_numbers(const std::string& key, const std::string& value);
/// Whole numbers separated by commas, likewise.
std::vector<std::int64_t> option_integers(const std::string& key, const std::string& value);
/// One of `allowed` exactly; otherwise throws naming the key and the allowed words.
std::string option_choice(const std::string& key, const std::string& value, const std::vector<std::string>& allowed);

}  // namespace sonar::live
