#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  m.def("predict_delay", &motion::predict_delay, py::arg("pings"), py::arg("measurement"),
        py::arg("y"), py::arg("z"), py::arg("sound_speed"),
        "Model delay tau_2 - tau_1 of a measurement for a trajectory.");
}
