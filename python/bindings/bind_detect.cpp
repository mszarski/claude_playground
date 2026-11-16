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
      py::arg("power"), py::arg("pfa") = 1e-4, py::arg("guard") = 4, py::arg("train") = 32,
      py::arg("backend") = Backend::Host, py::arg("return_ratio") = false, py::arg("device") = 0,
      "Cell-averaging CFAR along range on a power map [n_bearing, n_range]. Returns the\n"
      "boolean detection map, and with return_ratio the power over its local background.");
}
