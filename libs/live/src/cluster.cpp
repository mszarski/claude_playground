#include <cmath>

#include "sonar/live/detections.hpp"

namespace sonar::live {

std::vector<Detection> cluster(const unsigned char* detections, const float* power, const float* ratio,
                               std::int64_t n_rows, std::int64_t n_cols, const std::vector<double>& ranges,
                               const std::vector<double>& bearings) {
  std::vector<Detection> out;
  std::vector<unsigned char> seen(static_cast<std::size_t>(n_rows * n_cols), 0);
  std::vector<std::int64_t> stack;
  for (std::int64_t start = 0; start < n_rows * n_cols; ++start) {
    if (!detections[start] || seen[static_cast<std::size_t>(start)]) continue;
    // One group: everything reachable through touching cells, diagonals included.
    double w_sum = 0.0, r_sum = 0.0, b_sum = 0.0, best = 0.0;
    std::int64_t cells = 0;
    stack.assign(1, start);
    seen[static_cast<std::size_t>(start)] = 1;
    while (!stack.empty()) {
      const std::int64_t i = stack.back();
      stack.pop_back();
      const std::int64_t row = i / n_cols, col = i % n_cols;
      const double w = static_cast<double>(power[i]);
      w_sum += w;
      r_sum += w * ranges[static_cast<std::size_t>(col)];
      b_sum += w * bearings[static_cast<std::size_t>(row)];
      if (static_cast<double>(ratio[i]) > best) best = static_cast<double>(ratio[i]);
      ++cells;
      for (std::int64_t dr = -1; dr <= 1; ++dr) {
        for (std::int64_t dc = -1; dc <= 1; ++dc) {
          const std::int64_t r2 = row + dr, c2 = col + dc;
          if (r2 < 0 || r2 >= n_rows || c2 < 0 || c2 >= n_cols) continue;
          const std::int64_t j = r2 * n_cols + c2;
          if (detections[j] && !seen[static_cast<std::size_t>(j)]) {
            seen[static_cast<std::size_t>(j)] = 1;
            stack.push_back(j);
          }
        }
      }
    }
    out.push_back({r_sum / w_sum, b_sum / w_sum, 10.0 * std::log10(best), cells});
  }
  return out;
}

}  // namespace sonar::live
