#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "sonar/detect/cfar.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

void bind_detect(py::module_& m) {
  m.def("cfar_threshold_factor", &detect::cfar_threshold_factor, py::arg("pfa"), py::arg("n_train"));
  m.def(
      "cfar",
      [](const py::array_t<float, py::array::c_style | py::array::forcecast>& power, double pfa, int guard,
         int train, Backend backend, bool return_ratio, int device) -> py::object {
        if (power.ndim() != 2) throw std::invalid_argument("power must be [n_bearing, n_range]");
        detect::CfarOptions o;
        o.pfa = pfa;
        o.guard = guard;
        o.train = train;
        const std::vector<py::ssize_t> shape{power.shape(0), power.shape(1)};
        py::array_t<bool> det(shape);
        py::array_t<float> ratio(return_ratio ? shape : std::vector<py::ssize_t>{0, 0});
        {
          const float* p = power.data();
          auto* d = reinterpret_cast<unsigned char*>(det.mutable_data());
          float* r = return_ratio ? ratio.mutable_data() : nullptr;
          const auto rows = static_cast<std::int64_t>(shape[0]);
          const auto cols = static_cast<std::int64_t>(shape[1]);
          py::gil_scoped_release release;
          detect::cfar(backend, p, rows, cols, o, d, r, device);
        }
        if (return_ratio) return py::make_tuple(det, ratio);
        return std::move(det);
      },
      py::arg("power"), py::arg("pfa") = detect::CfarOptions{}.pfa, py::arg("guard") = detect::CfarOptions{}.guard,
      py::arg("train") = detect::CfarOptions{}.train,
      py::arg("backend") = Backend::Host, py::arg("return_ratio") = false, py::arg("device") = 0,
      "Cell-averaging CFAR along range on a power map [n_bearing, n_range]. Returns the\n"
      "boolean detection map, and with return_ratio the power over its local background.\n"
      "`guard` and `train` are CELLS of the map on each side; the defaults assume a cell is about one\n"
      "range resolution cell. For a map sampled finer than the resolution use cfar_window(), or the\n"
      "guard lies inside the target and it masks itself.");
  m.def(
      "cfar_window",
      [](double range_step, double resolution, double guard_resolutions, double train_resolutions, double guard_m,
         double train_m) {
        const detect::CfarOptions o =
            detect::cfar_window(range_step, resolution, {guard_resolutions, train_resolutions, guard_m, train_m});
        return py::make_tuple(o.guard, o.train);
      },
      py::arg("range_step"), py::arg("resolution"),
      py::arg("guard_resolutions") = detect::CfarWindow{}.guard_resolutions,
      py::arg("train_resolutions") = detect::CfarWindow{}.train_resolutions, py::arg("guard_m") = 0.0,
      py::arg("train_m") = 0.0,
      "(guard, train) cells for a map stepping `range_step` metres in range, for a range resolution of\n"
      "`resolution` metres (c / 2B). The window is `guard_resolutions` and `train_resolutions` resolution\n"
      "cells on each side, or `guard_m` / `train_m` metres when those are > 0 (the guard is at least one\n"
      "resolution cell). A map sampled at the resolution gets the defaults of cfar(): (4, 32).");
  m.attr("CFAR_DEFAULT_PFA") = detect::CfarOptions{}.pfa;
  m.attr("CFAR_DEFAULT_GUARD") = detect::CfarOptions{}.guard;
  m.attr("CFAR_DEFAULT_TRAIN") = detect::CfarOptions{}.train;
}
