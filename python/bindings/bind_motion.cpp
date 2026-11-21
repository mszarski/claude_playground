#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <stdexcept>

#include "sonar/motion/delay_estimation.hpp"
#include "sonar/motion/dpc_solver.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

void bind_motion(py::module_& m) {
  using motion::DelayMeasurement;
  using motion::PingNav;
  using motion::Solution;
  using motion::SolveOptions;

  py::class_<PingNav>(m, "PingNav")
      .def(py::init<>())
      .def_readwrite("time", &PingNav::time)
      .def_readwrite("x", &PingNav::x)
      .def_readwrite("surge", &PingNav::surge)
      .def_readwrite("y", &PingNav::y)
      .def_readwrite("z", &PingNav::z)
      .def_readwrite("rpy", &PingNav::rpy)
      .def_readwrite("dvl_altitude", &PingNav::dvl_altitude);

  py::class_<DelayMeasurement>(m, "DelayMeasurement")
      .def(py::init<>())
      .def_readwrite("ping", &DelayMeasurement::ping)
      .def_readwrite("tx_lever_1", &DelayMeasurement::tx_lever_1)
      .def_readwrite("rx_lever_1", &DelayMeasurement::rx_lever_1)
      .def_readwrite("tx_lever_2", &DelayMeasurement::tx_lever_2)
      .def_readwrite("rx_lever_2", &DelayMeasurement::rx_lever_2)
      .def_readwrite("t_ref", &DelayMeasurement::t_ref)
      .def_readwrite("footprint", &DelayMeasurement::footprint)
      .def_readwrite("delay", &DelayMeasurement::delay)
      .def_readwrite("weight", &DelayMeasurement::weight);

  py::class_<SolveOptions>(m, "MotionSolveOptions")
      .def(py::init<>())
      .def_readwrite("sound_speed", &SolveOptions::sound_speed)
      .def_readwrite("lambda_smooth", &SolveOptions::lambda_smooth)
      .def_readwrite("lambda_dvl", &SolveOptions::lambda_dvl)
      .def_readwrite("huber", &SolveOptions::huber)
      .def_readwrite("residual_scale", &SolveOptions::residual_scale)
      .def_readwrite("smooth_scale", &SolveOptions::smooth_scale)
      .def_readwrite("dvl_scale", &SolveOptions::dvl_scale)
      .def_readwrite("use_weights", &SolveOptions::use_weights)
      .def_readwrite("max_iterations", &SolveOptions::max_iterations);

  py::class_<Solution>(m, "MotionSolution")
      .def_readonly("y", &Solution::y)
      .def_readonly("z", &Solution::z)
      .def_readonly("vy", &Solution::vy)
      .def_readonly("vz", &Solution::vz)
      .def_readonly("initial_cost", &Solution::initial_cost)
      .def_readonly("final_cost", &Solution::final_cost)
      .def_readonly("iterations", &Solution::iterations)
      .def_readonly("converged", &Solution::converged)
      .def_readonly("report", &Solution::report);

  m.def("solve_motion", &motion::solve, py::arg("pings"), py::arg("measurements"), py::arg("options"),
        "Minimise ASASIN Eq. 8 for sway and heave with Ceres (Levenberg-Marquardt, autodiff).");
  using Samples = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>;
  m.def(
      "estimate_delays",
      [](const Samples& a, const Samples& b, const std::vector<std::int32_t>& channel_a,
         const std::vector<std::int32_t>& channel_b, const std::vector<std::int64_t>& centres, int half, int max_lag,
         double sample_rate, double center_frequency, Backend backend, int device) {
        if (a.ndim() != 2 || b.ndim() != 2 || a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
          throw std::invalid_argument("estimate_delays: a and b must be [n_channels, n_samples] of the same shape");
        }
        std::vector<motion::DelayEstimate> est;
        {
          py::gil_scoped_release release;
          est = motion::estimate_delays(a.data(), b.data(), a.shape(0), a.shape(1), channel_a, channel_b, centres,
                                        half, max_lag, sample_rate, center_frequency, backend, device);
        }
        const auto n = static_cast<py::ssize_t>(est.size());
        py::array_t<double> coarse(n), fine(n), coherence(n);
        py::array_t<bool> valid(n);
        for (py::ssize_t i = 0; i < n; ++i) {
          const auto& e = est[static_cast<std::size_t>(i)];
          coarse.mutable_at(i) = e.coarse;
          fine.mutable_at(i) = e.fine;
          coherence.mutable_at(i) = e.coherence;
          valid.mutable_at(i) = e.valid;
        }
        py::dict out;
        out["coarse"] = coarse;
        out["fine"] = fine;
        out["coherence"] = coherence;
        out["valid"] = valid;
        return out;
      },
      py::arg("a"), py::arg("b"), py::arg("channel_a"), py::arg("channel_b"), py::arg("centres"), py::arg("half"),
      py::arg("max_lag"), py::arg("sample_rate"), py::arg("center_frequency"), py::arg("backend") = Backend::Host,
      py::arg("device") = 0,
      "Coarse delay, fine delay, coherence and validity for every (channel pair, window centre), pair-major: "
      "the C++ port of gpusonar.motion.correlate_window.");
  m.def("predict_delay", &motion::predict_delay, py::arg("pings"), py::arg("measurement"),
        py::arg("y"), py::arg("z"), py::arg("sound_speed"),
        "Model delay tau_2 - tau_1 of a measurement for a trajectory.");
}
