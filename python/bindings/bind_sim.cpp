#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/sim/synth.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrC = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

Vec3d row3(const ArrD& a, py::ssize_t i) {
  auto v = a.unchecked<2>();
  return {v(i, 0), v(i, 1), v(i, 2)};
}

void require_n3(const ArrD& a, py::ssize_t n, const char* name) {
  if (a.ndim() != 2 || a.shape(1) != 3 || a.shape(0) != n) {
    throw std::invalid_argument(std::string(name) + " must have shape [n_paths, 3]");
  }
}

/// Build the path list from arrays. `q`/`p` are [P, 3]; velocities may be None (static).
std::vector<sim::SynthPath> make_paths(const ArrD& q, const ArrD& p, const ArrC& amplitude,
                                       const py::object& q_velocity, const py::object& p_velocity) {
  if (q.ndim() != 2 || q.shape(1) != 3) throw std::invalid_argument("q must have shape [n_paths, 3]");
  const py::ssize_t n = q.shape(0);
  require_n3(p, n, "p");
  if (amplitude.ndim() != 1 || amplitude.shape(0) != n) {
    throw std::invalid_argument("amplitude must have shape [n_paths]");
  }
  const bool has_vq = !q_velocity.is_none();
  const bool has_vp = !p_velocity.is_none();
  ArrD vq, vp;
  if (has_vq) {
    vq = ArrD::ensure(q_velocity);
    require_n3(vq, n, "q_velocity");
  }
  if (has_vp) {
    vp = ArrD::ensure(p_velocity);
    require_n3(vp, n, "p_velocity");
  }
  auto amp = amplitude.unchecked<1>();
  std::vector<sim::SynthPath> paths(static_cast<std::size_t>(n));
  for (py::ssize_t i = 0; i < n; ++i) {
    auto& path = paths[static_cast<std::size_t>(i)];
    path.q = row3(q, i);
    path.p = row3(p, i);
    if (has_vq) path.q_velocity = row3(vq, i);
    if (has_vp) path.p_velocity = row3(vp, i);
    path.amp_re = static_cast<float>(amp(i).real());
    path.amp_im = static_cast<float>(amp(i).imag());
  }
  return paths;
}

}  // namespace

void bind_sim(py::module_& m) {
  m.def(
      "synthesize_into",
      [](PingRecord& record, const ArrD& q, const ArrD& p, const ArrC& amplitude,
         const py::object& q_velocity, const py::object& p_velocity,
         const py::object& absorption, bool spreading, Backend backend) {
        const auto paths = make_paths(q, p, amplitude, q_velocity, p_velocity);
        sim::SynthParams params = sim::params_from_record(record);
        params.spreading = spreading;
        if (!absorption.is_none()) {
          // (f_min, f_max, table of kAbsorptionTableSize values in dB/m)
          const auto t = absorption.cast<py::tuple>();
          params.absorption_f_min = t[0].cast<double>();
          params.absorption_f_max = t[1].cast<double>();
          const auto table = t[2].cast<std::vector<double>>();
          if (table.size() != sim::kAbsorptionTableSize) {
            throw std::invalid_argument("absorption table must have " +
                                        std::to_string(sim::kAbsorptionTableSize) + " entries");
          }
          for (int i = 0; i < sim::kAbsorptionTableSize; ++i) params.absorption_db_per_m[i] = table[i];
        }
        py::gil_scoped_release release;
        sim::synthesize_into(backend, record, paths, &params);
      },
      py::arg("record"), py::arg("q"), py::arg("p"), py::arg("amplitude"),
      py::arg("q_velocity") = py::none(), py::arg("p_velocity") = py::none(),
      py::arg("absorption") = py::none(), py::arg("spreading") = true,
      py::arg("backend") = Backend::Host,
      "Add the echoes of two-way paths into record.samples. The record supplies geometry,\n"
      "waveform, sampling and board clocks; its samples must already be allocated.");
  m.attr("ABSORPTION_TABLE_SIZE") = sim::kAbsorptionTableSize;
}
