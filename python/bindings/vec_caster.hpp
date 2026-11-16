// pybind11 conversion for sonar::Vec3d: accepts any length-3 sequence, returns a tuple.
// Returned values are copies, so assign whole vectors: `pose.position = (x, y, z)`.
#pragma once

#include <pybind11/pybind11.h>

#include "sonar/core/vec.hpp"

namespace pybind11::detail {

template <>
struct type_caster<sonar::Vec3d> {
 public:
  PYBIND11_TYPE_CASTER(sonar::Vec3d, const_name("tuple[float, float, float]"));

  bool load(handle src, bool /*convert*/) {
    if (!src || !PySequence_Check(src.ptr())) return false;
    auto seq = reinterpret_borrow<sequence>(src);
    if (seq.size() != 3) return false;
    try {
      value = sonar::Vec3d(seq[0].cast<double>(), seq[1].cast<double>(), seq[2].cast<double>());
    } catch (const cast_error&) {
      return false;
    }
    return true;
  }

  static handle cast(const sonar::Vec3d& v, return_value_policy /*policy*/, handle /*parent*/) {
    return make_tuple(v.x, v.y, v.z).release();
  }
};

}  // namespace pybind11::detail
