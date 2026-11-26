// Bindings of the simulator parts ported from NumPy: the eigenray providers without a
// ray tracer (sonar/sim/paths.hpp) and scattering-function reverberation
// (sonar/sim/reverb.hpp). python/gpusonar/sim/engine.py chooses between these and the
// NumPy implementations.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>

#include "sonar/sim/paths.hpp"
#include "sonar/sim/reverb.hpp"
#include "sonar/sim/reverb_batched.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrC = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
using ArrB = py::array_t<bool, py::array::c_style | py::array::forcecast>;
using cd = std::complex<double>;

/// A vector as a NumPy array of element type `As` (the same size as T) without a copy:
/// the array keeps the vector alive.
template <typename T, typename As = T>
py::array to_array(std::vector<T>&& v, const std::vector<py::ssize_t>& shape) {
  static_assert(sizeof(T) == sizeof(As), "element types must have the same size");
  auto* held = new std::vector<T>(std::move(v));
  const py::capsule owner(held, [](void* ptr) { delete static_cast<std::vector<T>*>(ptr); });
  return py::array(py::dtype::of<As>(), shape, held->data(), owner);
}

void require_points(const ArrD& a, const char* name) {
  if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument(std::string(name) + " must have shape [n, 3]");
}

/// The Python EigenraySet's fields, as a dict of arrays.
py::dict set_dict(sim::EigenraySet&& e) {
  const py::ssize_t s = e.n_points, k = e.n_rays;
  py::dict d;
  d["delay"] = to_array(std::move(e.delay), {s, k});
  d["amplitude"] = to_array(std::move(e.amplitude), {s, k});
  d["launch"] = to_array(std::move(e.launch), {s, k, 3});
  d["arrival"] = to_array(std::move(e.arrival), {s, k, 3});
  d["n_surface"] = to_array(std::move(e.n_surface), {s, k});
  d["n_bottom"] = to_array(std::move(e.n_bottom), {s, k});
  d["valid"] = to_array<std::uint8_t, bool>(std::move(e.valid), {s, k});  // 0 or 1 in every byte
  d["sound_speed"] = e.sound_speed;
  return d;
}

/// The fields two_way_paths reads, as a view of arrays [S, K] (directions [S, K, 3]).
sim::EigenrayView set_view(const ArrD& delay, const ArrC& amplitude, const ArrD& launch, const ArrD& arrival,
                           const ArrB& valid, double sound_speed) {
  if (delay.ndim() != 2) throw std::invalid_argument("delay must have shape [points, rays]");
  const py::ssize_t s = delay.shape(0), k = delay.shape(1);
  const auto same = [&](const py::array& a) { return a.ndim() == 2 && a.shape(0) == s && a.shape(1) == k; };
  const auto dirs = [&](const py::array& a) {
    return a.ndim() == 3 && a.shape(0) == s && a.shape(1) == k && a.shape(2) == 3;
  };
  if (!same(amplitude) || !same(valid) || !dirs(launch) || !dirs(arrival)) {
    throw std::invalid_argument("eigenray arrays must share the shape [points, rays]");
  }
  static_assert(sizeof(bool) == 1, "numpy bool arrays are read as bytes");
  return {s,           k,              sound_speed,    delay.data(), amplitude.data(),
          launch.data(), arrival.data(), reinterpret_cast<const std::uint8_t*>(valid.data())};
}

/// The three reverberation entry points below that take no sound speed (which rows are
/// active, the sector standard deviations, the final transform) never read it, but the
/// shared parameter check in libs/sim/src/reverb.cpp demands a positive value. This is that
/// placeholder, named so that it cannot be mistaken for a physical default (audit S-4): an
/// entry point that starts to use the sound speed must take it from its caller instead.
constexpr double kSoundSpeedNotUsed = 1500.0;

/// A bottom half-space from propagation._fluid_bottom's tuple: (speed, density, attenuation, water_density) for a
/// fluid, or those and (shear_speed, shear_attenuation) for an elastic solid (A8).
sim::FluidBottom fluid_bottom(const py::object& bottom) {
  const auto t = bottom.cast<py::tuple>();
  if (t.size() != 4 && t.size() != 6) {
    throw std::invalid_argument("a bottom is (speed, density, attenuation, water_density[, shear_speed, shear_attenuation])");
  }
  sim::FluidBottom b{t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<double>()};
  if (t.size() == 6) {
    b.shear_speed = t[4].cast<double>();
    b.shear_attenuation = t[5].cast<double>();
  }
  return b;
}

/// An image-source configuration from the binding's arguments. Nothing is defaulted here
/// that the C++ struct does not default (audit S-1, S-3): a bottom without a coefficient
/// or a sediment, and a missing max_order, are errors raised by sim::check_config.
sim::ImageSourceConfig image_config(double sound_speed, const py::object& surface_z, const py::object& bottom_z,
                                    cd surface_coefficient, const py::object& bottom_coefficient,
                                    const py::object& max_order,
                                    const std::vector<std::tuple<int, double, cd>>& walls, const py::object& bottom) {
  sim::ImageSourceConfig cfg;
  cfg.sound_speed = sound_speed;
  cfg.has_surface = !surface_z.is_none();
  if (cfg.has_surface) cfg.surface_z = surface_z.cast<double>();
  cfg.has_bottom = !bottom_z.is_none();
  if (cfg.has_bottom) cfg.bottom_z = bottom_z.cast<double>();
  cfg.surface_coefficient = surface_coefficient;
  if (!bottom_coefficient.is_none()) cfg.bottom_coefficient = bottom_coefficient.cast<cd>();
  if (!max_order.is_none()) cfg.max_order = max_order.cast<int>();
  for (const auto& [axis, position, coefficient] : walls) cfg.walls.push_back({axis, position, coefficient});
  cfg.has_fluid_bottom = !bottom.is_none();
  if (cfg.has_fluid_bottom) cfg.fluid_bottom = fluid_bottom(bottom);
  return cfg;
}

/// The first `n` rows of an array that was sized for more.
template <typename T>
py::array_t<T> first_rows(const py::array_t<T>& a, py::ssize_t n, py::ssize_t width) {
  if (a.shape(0) == n) return a;
  const std::vector<py::ssize_t> shape = width > 0 ? std::vector<py::ssize_t>{n, width} : std::vector<py::ssize_t>{n};
  py::array_t<T> out(shape);
  if (n > 0) std::memcpy(out.mutable_data(), a.data(), static_cast<std::size_t>(n * (width > 0 ? width : 1)) * sizeof(T));
  return out;
}

}  // namespace

void bind_simcpp(py::module_& m) {
  m.def(
      "sim_free_field_eigenrays",
      [](double sound_speed, const Vec3d& source, const ArrD& points) {
        require_points(points, "points");
        sim::EigenraySet e;
        {
          py::gil_scoped_release release;
          e = sim::free_field_eigenrays(sound_speed, source, points.data(), points.shape(0));
        }
        return set_dict(std::move(e));
      },
      py::arg("sound_speed"), py::arg("source"), py::arg("points"),
      "Direct-path eigenrays from `source` to `points` [S, 3]: a dict of the EigenraySet fields.");

  m.def(
      "sim_image_source_eigenrays",
      [](double sound_speed, const Vec3d& source, const ArrD& points, const py::object& surface_z,
         const py::object& bottom_z, cd surface_coefficient, const py::object& bottom_coefficient,
         const py::object& max_order, const std::vector<std::tuple<int, double, cd>>& walls,
         const py::object& bottom) {
        require_points(points, "points");
        const sim::ImageSourceConfig cfg = image_config(sound_speed, surface_z, bottom_z, surface_coefficient,
                                                        bottom_coefficient, max_order, walls, bottom);
        sim::EigenraySet e;
        {
          py::gil_scoped_release release;
          e = sim::image_source_eigenrays(cfg, source, points.data(), points.shape(0));
        }
        return set_dict(std::move(e));
      },
      py::arg("sound_speed"), py::arg("source"), py::arg("points"), py::arg("surface_z") = 0.0,
      py::arg("bottom_z") = py::none(), py::arg("surface_coefficient") = cd{-1.0, 0.0},
      py::arg("bottom_coefficient") = py::none(), py::arg("max_order") = py::none(),
      py::arg("walls") = std::vector<std::tuple<int, double, cd>>{}, py::arg("bottom") = py::none(),
      "Image-source eigenrays (flat surface and bottom, optional walls [(axis, position, coefficient)] and a "
      "fluid bottom (speed, density, attenuation, water_density)): a dict of the EigenraySet fields.\n"
      "`max_order` (bounces along one path) is required, and a bottom needs `bottom_coefficient` or `bottom`:\n"
      "neither has a default. Paths with more than `max_order` bounces are not computed\n"
      "(sim_image_series_truncation gives what that leaves out).");

  m.def(
      "sim_plane_image_eigenrays",
      [](double sound_speed, const Vec3d& source, const ArrD& points, const py::list& planes, int max_order,
         int wall_order, cd bottom_coefficient, const py::object& bottom) {
        require_points(points, "points");
        sim::PlaneImageConfig cfg;
        cfg.sound_speed = sound_speed;
        cfg.max_order = max_order;
        cfg.wall_order = wall_order;
        cfg.bottom_coefficient = bottom_coefficient;
        cfg.has_fluid_bottom = !bottom.is_none();
        if (cfg.has_fluid_bottom) cfg.fluid_bottom = fluid_bottom(bottom);
        for (const py::handle item : planes) {
          // (kind, point, normal, coefficient or None, [(axis, lower, upper), ...])
          const auto t = item.cast<py::tuple>();
          if (t.size() != 5) throw std::invalid_argument("a plane is (kind, point, normal, coefficient, bounds)");
          sim::Plane p;
          const auto kind = t[0].cast<std::string>();
          if (kind == "surface") {
            p.kind = sim::PlaneKind::Surface;
          } else if (kind == "bottom") {
            p.kind = sim::PlaneKind::Bottom;
          } else if (kind == "wall") {
            p.kind = sim::PlaneKind::Wall;
          } else {
            throw std::invalid_argument("plane kind must be surface, bottom or wall, not " + kind);
          }
          p.point = t[1].cast<Vec3d>();
          p.normal = t[2].cast<Vec3d>();  // already unit (`_Plane` normalises it): taken as given
          if (!t[3].is_none()) p.coefficient = t[3].cast<cd>();
          for (const auto& [axis, lower, upper] : t[4].cast<std::vector<std::tuple<int, double, double>>>()) {
            if (axis < 0 || axis > 2) throw std::invalid_argument("a bound's axis must be 0, 1 or 2");
            p.has_bound[axis] = true;
            p.lower[axis] = lower;
            p.upper[axis] = upper;
          }
          cfg.planes.push_back(p);
        }
        sim::EigenraySet e;
        {
          py::gil_scoped_release release;
          e = sim::plane_image_eigenrays(cfg, source, points.data(), points.shape(0));
        }
        return set_dict(std::move(e));
      },
      py::arg("sound_speed"), py::arg("source"), py::arg("points"), py::arg("planes"), py::arg("max_order"),
      py::arg("wall_order"), py::arg("bottom_coefficient"), py::arg("bottom") = py::none(),
      "Image-source eigenrays of the general construction (propagation._PlaneImages): `planes` a list of\n"
      "(kind 'surface' | 'bottom' | 'wall', point [3], unit normal [3] into the water, coefficient or None for a\n"
      "bottom, bounds [(axis, lower, upper), ...]); `max_order` surface and bottom bounces and `wall_order` wall\n"
      "bounces a path; a bottom bounce takes the Rayleigh coefficient of `bottom` (speed, density, attenuation,\n"
      "water_density) at its own grazing angle, or `bottom_coefficient`. A dict of the EigenraySet fields.");

  m.def(
      "sim_image_series_truncation",
      [](double sound_speed, double source_z, double receiver_z, double range, const py::object& surface_z,
         const py::object& bottom_z, cd surface_coefficient, const py::object& bottom_coefficient,
         const py::object& max_order, const py::object& bottom) {
        return sim::image_series_truncation(
            image_config(sound_speed, surface_z, bottom_z, surface_coefficient, bottom_coefficient, max_order, {},
                         bottom),
            source_z, receiver_z, range);
      },
      py::arg("sound_speed"), py::arg("source_z"), py::arg("receiver_z"), py::arg("range"),
      py::arg("surface_z") = 0.0, py::arg("bottom_z") = py::none(),
      py::arg("surface_coefficient") = cd{-1.0, 0.0}, py::arg("bottom_coefficient") = py::none(),
      py::arg("max_order") = py::none(), py::arg("bottom") = py::none(),
      "What cutting the image series at `max_order` leaves out: the largest amplitude among the rays with\n"
      "max_order + 1 bounces between depths `source_z` and `receiver_z` a horizontal `range` apart, relative\n"
      "to the direct path's (0 when the series has ended).");

  m.def(
      "sim_rayleigh_reflection",
      [](const ArrD& grazing, double c_water, double speed, double density, double attenuation, double water_density,
         double shear_speed, double shear_attenuation) {
        py::array_t<cd> out(std::vector<py::ssize_t>(grazing.shape(), grazing.shape() + grazing.ndim()));
        const sim::FluidBottom b{speed, density, attenuation, water_density, shear_speed, shear_attenuation};
        for (py::ssize_t i = 0; i < grazing.size(); ++i) {
          out.mutable_data()[i] = sim::rayleigh_reflection(grazing.data()[i], c_water, b);
        }
        return out;
      },
      py::arg("grazing"), py::arg("c_water"), py::arg("speed"), py::arg("density"), py::arg("attenuation") = 0.0,
      py::arg("water_density") = 1.0, py::arg("shear_speed") = 0.0, py::arg("shear_attenuation") = 0.0,
      "Rayleigh reflection coefficient of a fluid half-space at grazing angles (rad); with shear_speed > 0 (m/s) and\n"
      "shear_attenuation (dB/wavelength) that of an elastic one, J&R (9.34)-(9.40).");

  m.def(
      "sim_snell_crossing",
      [](const Vec3d& source, const ArrD& points, double c_water, double c_sediment, double interface_z) {
        require_points(points, "points");
        const py::ssize_t s = points.shape(0);
        py::array_t<double> time(s);
        py::array_t<double> cross({s, py::ssize_t{3}});
        {
          py::gil_scoped_release release;
          sim::snell_crossing(source, points.data(), s, c_water, c_sediment, interface_z, time.mutable_data(),
                              cross.mutable_data());
        }
        return py::make_tuple(time, cross);
      },
      py::arg("source"), py::arg("points"), py::arg("c_water"), py::arg("c_sediment"), py::arg("interface_z"),
      "Least-time paths through a flat water-sediment interface: (travel time [S], crossing points [S, 3]).");

  m.def(
      "sim_sediment_eigenrays",
      [](double sound_speed, double sediment_speed, double interface_z, cd transmission, const Vec3d& source,
         const ArrD& points, const py::object& bottom, const py::object& frequency, bool evanescent) {
        require_points(points, "points");
        sim::SedimentConfig cfg;
        cfg.sound_speed = sound_speed;
        cfg.sediment_speed = sediment_speed;
        cfg.interface_z = interface_z;
        cfg.transmission = transmission;
        cfg.fluid = !bottom.is_none();
        if (cfg.fluid) {
          // (speed, density, attenuation, water_density), as for an image-source fluid bottom
          const auto t = bottom.cast<std::tuple<double, double, double, double>>();
          cfg.bottom = {std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)};
          if (!frequency.is_none()) cfg.frequency = frequency.cast<double>();
          cfg.evanescent = evanescent;
        }
        sim::EigenraySet e;
        {
          py::gil_scoped_release release;
          e = sim::sediment_eigenrays(cfg, source, points.data(), points.shape(0));
        }
        return set_dict(std::move(e));
      },
      py::arg("sound_speed"), py::arg("sediment_speed"), py::arg("interface_z"), py::arg("transmission"),
      py::arg("source"), py::arg("points"), py::arg("bottom") = py::none(), py::arg("frequency") = py::none(), py::arg("evanescent") = false,
      "One refracted eigenray per point below a flat sediment interface: a dict of the EigenraySet fields. With "
      "`bottom` (speed, density, attenuation dB/wavelength, water_density) and `frequency` (Hz): the opt-in fluid "
      "penetration (transmission coefficient, reciprocity factor, attenuation in the sediment; `evanescent`: the evanescent field below the critical angle).");

  m.def(
      "sim_two_way_paths",
      [](const ArrD& out_delay, const ArrC& out_amplitude, const ArrD& out_launch, const ArrD& out_arrival,
         const ArrB& out_valid, double out_sound_speed, const ArrD& back_delay, const ArrC& back_amplitude,
         const ArrD& back_launch, const ArrD& back_arrival, const ArrB& back_valid, double back_sound_speed,
         const Vec3d& tx_position, const Vec3d& rx_reference, const ArrC& amplitudes, const py::object& velocities,
         const py::object& lambert, const py::object& normals, const py::object& out_range_rate,
         const py::object& back_range_rate) -> py::tuple {
        const sim::EigenrayView out = set_view(out_delay, out_amplitude, out_launch, out_arrival, out_valid, out_sound_speed);
        const sim::EigenrayView back =
            set_view(back_delay, back_amplitude, back_launch, back_arrival, back_valid, back_sound_speed);
        const py::ssize_t s = out.n_points;
        if (back.n_points != s) throw std::invalid_argument("the two eigenray sets must cover the same points");
        if (amplitudes.ndim() != 1 || amplitudes.shape(0) != s) throw std::invalid_argument("amplitudes must have shape [points]");
        ArrD vel, nrm;
        ArrB lam;
        if (!velocities.is_none()) {
          vel = ArrD::ensure(velocities);
          if (!vel || vel.ndim() != 2 || vel.shape(0) != s || vel.shape(1) != 3) throw std::invalid_argument("velocities must have shape [points, 3]");
        }
        if (!lambert.is_none()) {
          lam = ArrB::ensure(lambert);
          if (!lam || lam.ndim() != 1 || lam.shape(0) != s) throw std::invalid_argument("lambert must have shape [points]");
        }
        if (!normals.is_none()) {
          nrm = ArrD::ensure(normals);
          if (!nrm || nrm.ndim() != 2 || nrm.shape(0) != s || nrm.shape(1) != 3) throw std::invalid_argument("normals must have shape [points, 3]");
        }
        const double* v_in = velocities.is_none() ? nullptr : vel.data();
        const bool moving = sim::any_moving(v_in, s);
        // Room for every pair; nearly all are kept.
        const py::ssize_t room = s * out.n_rays * back.n_rays;
        py::array_t<double> q({room, py::ssize_t{3}}), p({room, py::ssize_t{3}});
        py::array_t<cd> amp(room);
        // Per-ray range rates (either leg): each pair gets a velocity for each of its ends.
        if (!out_range_rate.is_none() || !back_range_rate.is_none()) {
          ArrD rate_out, rate_back;
          sim::RangeRates rates;
          if (!out_range_rate.is_none()) {
            rate_out = ArrD::ensure(out_range_rate);
            if (!rate_out || rate_out.ndim() != 2 || rate_out.shape(0) != s || rate_out.shape(1) != out.n_rays) {
              throw std::invalid_argument("out_range_rate must have the shape of out_delay");
            }
            rates.out = rate_out.data();
          }
          if (!back_range_rate.is_none()) {
            rate_back = ArrD::ensure(back_range_rate);
            if (!rate_back || rate_back.ndim() != 2 || rate_back.shape(0) != s || rate_back.shape(1) != back.n_rays) {
              throw std::invalid_argument("back_range_rate must have the shape of back_delay");
            }
            rates.back = rate_back.data();
          }
          py::array_t<double> vq({room, py::ssize_t{3}}), vp({room, py::ssize_t{3}});
          py::ssize_t n = 0;
          {
            py::gil_scoped_release release;
            n = sim::two_way_paths_doppler(
                out, back, tx_position, rx_reference, amplitudes.data(), moving ? v_in : nullptr,
                lambert.is_none() ? nullptr : reinterpret_cast<const std::uint8_t*>(lam.data()),
                normals.is_none() ? nullptr : nrm.data(), rates, q.mutable_data(), p.mutable_data(), amp.mutable_data(),
                vq.mutable_data(), vp.mutable_data());
          }
          return py::make_tuple(first_rows(q, n, 3), first_rows(p, n, 3), first_rows(amp, n, 0), first_rows(vq, n, 3),
                                first_rows(vp, n, 3));
        }
        py::array_t<double> v_out({moving ? room : py::ssize_t{0}, py::ssize_t{3}});
        py::ssize_t n = 0;
        {
          py::gil_scoped_release release;
          n = sim::two_way_paths(out, back, tx_position, rx_reference, amplitudes.data(), v_in,
                                 lambert.is_none() ? nullptr : reinterpret_cast<const std::uint8_t*>(lam.data()),
                                 normals.is_none() ? nullptr : nrm.data(), q.mutable_data(), p.mutable_data(),
                                 amp.mutable_data(), moving ? v_out.mutable_data() : nullptr);
        }
        py::object v = py::none();
        if (moving) v = first_rows(v_out, n, 3);
        return py::make_tuple(first_rows(q, n, 3), first_rows(p, n, 3), first_rows(amp, n, 0), v);
      },
      py::arg("out_delay"), py::arg("out_amplitude"), py::arg("out_launch"), py::arg("out_arrival"), py::arg("out_valid"),
      py::arg("out_sound_speed"), py::arg("back_delay"), py::arg("back_amplitude"), py::arg("back_launch"),
      py::arg("back_arrival"), py::arg("back_valid"), py::arg("back_sound_speed"), py::arg("tx_position"),
      py::arg("rx_reference"), py::arg("amplitudes"), py::arg("velocities") = py::none(), py::arg("lambert") = py::none(),
      py::arg("normals") = py::none(), py::arg("out_range_rate") = py::none(), py::arg("back_range_rate") = py::none(),
      "Kernel paths for every (outbound, return) eigenray pair: (q [P, 3], p [P, 3], amplitude [P], velocity [P, 3] or None).\n"
      "With `out_range_rate` [S, rays out] and/or `back_range_rate` [S, rays back] (path-length rates of the rays, m/s)\n"
      "the result is (q, p, amplitude, q_velocity [P, 3], p_velocity [P, 3]): each end's velocity is the scatterer's\n"
      "plus that leg's rate along its launch direction.");
  m.attr("SIM_PATHS_RANGE_RATE") = true;

  const auto reverb_params = [](double sample_rate, std::int64_t n_samples, double start_time, double time0,
                                double center_frequency, double sound_speed, std::int64_t n_azimuth,
                                std::int64_t decimation, std::int64_t n_low, const Vec3d& velocity, bool carrier,
                                bool real_output) {
    sim::ReverbParams p;
    p.sample_rate = sample_rate;
    p.n_samples = n_samples;
    p.start_time = start_time;
    p.time0 = time0;
    p.center_frequency = center_frequency;
    p.sound_speed = sound_speed;
    p.n_azimuth = n_azimuth;
    p.decimation = decimation;
    p.n_low = n_low;
    p.velocity = velocity;
    p.carrier = carrier;
    p.real_output = real_output;
    return p;
  };

  // `direction` [rows, 3] and `doppler` [rows] (both or neither) make each row of the density a
  // component with its own arrival direction and Doppler shift (sonar/sim/reverb.hpp).
  const auto reverb_power = [](const ArrD& coarse, const ArrD& density, const std::optional<ArrD>& direction,
                               const std::optional<ArrD>& doppler) {
    if (coarse.ndim() != 1 || density.ndim() != 2 || density.shape(1) != coarse.shape(0)) {
      throw std::invalid_argument("density must have shape [rows, len(coarse)]");
    }
    sim::ReverbPower power{coarse.data(), coarse.shape(0), density.data(), density.shape(0)};
    if (direction.has_value() != doppler.has_value()) {
      throw std::invalid_argument("direction and doppler are given together");
    }
    if (direction) {
      if (direction->ndim() != 2 || direction->shape(0) != density.shape(0) || direction->shape(1) != 3 ||
          doppler->ndim() != 1 || doppler->shape(0) != density.shape(0)) {
        throw std::invalid_argument("direction must have shape [rows, 3] and doppler [rows]");
      }
      power.direction = direction->data();
      power.doppler = doppler->data();
    }
    return power;
  };

  m.def(
      "sim_reverb_active",
      [reverb_params, reverb_power](const ArrD& coarse, const ArrD& density, double sample_rate, double time0,
                                    std::int64_t n_azimuth, std::int64_t decimation, std::int64_t n_low,
                                    bool components) {
        sim::ReverbPower power = reverb_power(coarse, density, std::nullopt, std::nullopt);
        // Which rows have power does not depend on the directions: any non-null pointers
        // select component mode (the rows need not match n_azimuth).
        if (components) power.direction = power.doppler = density.data();
        const sim::ReverbParams p =
            reverb_params(sample_rate, 0, 0.0, time0, 0.0, kSoundSpeedNotUsed, n_azimuth, decimation, n_low, Vec3d{},
                          false, false);
        std::vector<std::uint8_t> active;
        {
          py::gil_scoped_release release;
          active = sim::reverb_active_rows(p, power);
        }
        py::array_t<bool> out(static_cast<py::ssize_t>(active.size()));
        for (std::size_t i = 0; i < active.size(); ++i) out.mutable_data()[i] = active[i] != 0;
        return out;
      },
      py::arg("coarse"), py::arg("density"), py::arg("sample_rate"), py::arg("time0"), py::arg("n_azimuth"),
      py::arg("decimation"), py::arg("n_low"), py::arg("components") = false,
      "Which rows of a reverberation power density [rows, len(coarse)] give a sector any power at the reduced "
      "rate: the sectors that take random numbers.");

  m.def(
      "sim_reverb_sigma",
      [reverb_params, reverb_power](const ArrD& coarse, const ArrD& density, double sample_rate, double time0,
                                    std::int64_t n_azimuth, std::int64_t decimation, std::int64_t n_low) {
        const sim::ReverbPower power = reverb_power(coarse, density, std::nullopt, std::nullopt);
        const sim::ReverbParams p =
            reverb_params(sample_rate, 0, 0.0, time0, 0.0, kSoundSpeedNotUsed, n_azimuth, decimation, n_low, Vec3d{},
                          false, false);
        std::vector<double> sigma;
        {
          py::gil_scoped_release release;
          sigma = sim::reverb_sigma(p, power);
        }
        return to_array(std::move(sigma), {density.shape(0), static_cast<py::ssize_t>(n_low)});
      },
      py::arg("coarse"), py::arg("density"), py::arg("sample_rate"), py::arg("time0"), py::arg("n_azimuth"),
      py::arg("decimation"), py::arg("n_low"),
      "Standard deviation of each azimuth sector's reverberation sequence at the reduced rate, [rows, n_low].");

  using ArrTotal = py::array_t<std::complex<float>, py::array::c_style>;
  m.def(
      "sim_reverb_accumulate",
      [reverb_params, reverb_power](ArrTotal& total, const ArrD& coarse, const ArrD& density, std::int64_t first,
                                    const ArrD& normals, const ArrC& pulse, const ArrD& offsets, double sample_rate,
                                    double time0, double center_frequency, double sound_speed,
                                    std::int64_t n_azimuth, std::int64_t decimation, const Vec3d& velocity,
                                    Backend backend, int device, const std::optional<ArrD>& direction,
                                    const std::optional<ArrD>& doppler,
                                    const std::optional<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>& rows,
                                    bool batched, bool total_from_host, bool total_to_host) {
        const sim::ReverbPower power = reverb_power(coarse, density, direction, doppler);
        require_points(offsets, "offsets");
        const py::ssize_t n_channels = offsets.shape(0);
        if (total.ndim() != 2 || total.shape(0) != n_channels || !total.writeable()) {
          throw std::invalid_argument("total must be a writeable complex64 array [channels, n_low]");
        }
        const py::ssize_t n_low = total.shape(1);
        if (normals.ndim() != 3 || normals.shape(1) != 2 || normals.shape(2) != n_low) {
          throw std::invalid_argument("normals must have shape [sectors, 2, n_low]");
        }
        if (pulse.ndim() != 1) throw std::invalid_argument("pulse must be one-dimensional");
        if (rows && (rows->ndim() != 1 || rows->shape(0) != normals.shape(0))) {
          throw std::invalid_argument("rows must have one entry per sector of normals");
        }
        sim::ReverbParams p = reverb_params(sample_rate, 0, 0.0, time0, center_frequency, sound_speed, n_azimuth,
                                            decimation, n_low, velocity, false, false);
        p.batched = batched;
        p.total_from_host = total_from_host;
        p.total_to_host = total_to_host;
        auto* data = reinterpret_cast<float*>(total.mutable_data());
        const std::int64_t* row_data = rows ? rows->data() : nullptr;
        py::gil_scoped_release release;
        sim::reverb_accumulate(backend, p, power, first, normals.data(), normals.shape(0), pulse.data(), pulse.shape(0),
                               offsets.data(), n_channels, data, device, row_data);
      },
      py::arg("total").noconvert(), py::arg("coarse"), py::arg("density"), py::arg("first"), py::arg("normals"),
      py::arg("pulse"), py::arg("offsets"), py::arg("sample_rate"), py::arg("time0"), py::arg("center_frequency"),
      py::arg("sound_speed"), py::arg("n_azimuth"), py::arg("decimation"), py::arg("velocity") = Vec3d{},
      py::arg("backend") = Backend::Host, py::arg("device") = 0, py::arg("direction") = py::none(),
      py::arg("doppler") = py::none(), py::arg("rows") = py::none(), py::arg("batched") = false,
      py::arg("total_from_host") = true, py::arg("total_to_host") = true,
      "Add a run of azimuth sectors of scattering-function reverberation into `total` (complex64 [channels, n_low], "
      "in place), from the power density [rows, len(coarse)]: `normals` [sectors, 2, n_low] are standard normal "
      "deviates for the sectors with power starting at the `first`-th. With `direction` [rows, 3] and `doppler` "
      "[rows], each row is a component with that arrival direction and frequency shift instead of an azimuth sector.\n"
      "`rows` [sectors] names the sector (the row, for components) of each entry of `normals`, so that the sectors "
      "with power are not searched for in every call (`first` is then not used). The CUDA backend runs the batched "
      "formulation (transforms, Doppler shifts and steering on the device); `batched=True` runs that formulation on "
      "the host. In the batched formulation `total_from_host=False` continues from the spectrum the previous call left "
      "on the backend with `total_to_host=False` (the array is then stale until a call brings it back): a "
      "generator call that adds its sectors in runs moves the spectrum to the device once and back once.");
  // What this build's reverberation generator can do (python/gpusonar/sim/reverb.py).
  m.attr("SIM_REVERB_COMPONENTS") = true;   // rows with their own direction and Doppler shift
  m.attr("SIM_REVERB_ROWS") = true;         // sim_reverb_accumulate(rows=...)
  m.attr("SIM_REVERB_BATCHED") = true;      // the batched formulation; sim_reverb_finish on a backend

  m.def(
      "sim_reverb_profile",
      [](bool reset) {
        const sim::ReverbProfile p = sim::reverb_profile();
        if (reset) sim::reverb_profile_reset();
        py::dict d;
        d["shape"] = p.shape;
        d["upload"] = p.upload;
        d["shift"] = p.shift;
        d["transform"] = p.transform;
        d["pulse"] = p.pulse;
        d["product"] = p.product;
        d["steer"] = p.steer;
        d["download"] = p.download;
        d["finish"] = p.finish;
        d["active"] = p.active;
        d["calls"] = p.calls;
        d["components"] = p.components;
        return d;
      },
      py::arg("reset") = false,
      "Seconds spent in each step of the reverberation generator since the last reset, with the number of calls of "
      "sim_reverb_accumulate and of components. Batched formulation: shape (host), upload, shift, transform, pulse, "
      "product, steer, download; reference host code: transform (the per-sector transforms on the CPU threads), "
      "steer; both: active (finding the sectors with power, when `rows` is not passed) and finish.");

  m.def(
      "sim_reverb_finish",
      [reverb_params](const ArrTotal& total, double sample_rate, std::int64_t n_samples, double start_time,
                      double time0, double center_frequency, std::int64_t decimation, bool carrier,
                      bool real_output, Backend backend, int device, bool batched) -> py::array {
        if (total.ndim() != 2) throw std::invalid_argument("total must have shape [channels, n_low]");
        const py::ssize_t n_channels = total.shape(0);
        sim::ReverbParams p = reverb_params(sample_rate, n_samples, start_time, time0, center_frequency,
                                            kSoundSpeedNotUsed, 1, decimation, total.shape(1), Vec3d{}, carrier,
                                            real_output);
        p.batched = batched;
        const std::vector<py::ssize_t> shape{n_channels, static_cast<py::ssize_t>(n_samples)};
        py::array out = real_output ? py::array(py::array_t<float>(shape)) : py::array(py::array_t<std::complex<float>>(shape));
        auto* data = static_cast<float*>(out.mutable_data());
        {
          py::gil_scoped_release release;
          sim::reverb_finish(p, reinterpret_cast<const float*>(total.data()), n_channels, data, backend, device);
        }
        return out;
      },
      py::arg("total"), py::arg("sample_rate"), py::arg("n_samples"), py::arg("start_time"), py::arg("time0"),
      py::arg("center_frequency"), py::arg("decimation"), py::arg("carrier") = false, py::arg("real_output") = false,
      py::arg("backend") = Backend::Host, py::arg("device") = 0, py::arg("batched") = false,
      "The reverberation of every channel in the record's window from the accumulated spectrum: [channels, "
      "n_samples] complex64, or float32 with real_output. On the CUDA backend the interpolation to the record's rate "
      "runs on the device; `batched=True` runs the device's formulation on the host.");
}
