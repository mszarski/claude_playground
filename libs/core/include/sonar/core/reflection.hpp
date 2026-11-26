#pragma once
// The flat-interface pressure reflection coefficient of an elastic half-space (rock), shared by the simulator
// (libs/sim paths.cpp `rayleigh_reflection`) and the live chain (libs/live raytrace.cpp `rayleigh_reflection`, which
// coverage, multipath_depth and multipath_rays call): one implementation, so the two give the same bits.
//
// Jackson and Richardson (2007) (9.34)-(9.40) with (9.23) (docs/references/jackson_richardson_2007/03 Sec. 2.3;
// docs/PHYSICS_NOTES.md Sec. 6.5), multiplied through by the two vertical wavenumbers so that no critical angle
// divides by zero: with m = rho_b / rho_w, n = c_w / c_p (complex with the loss), n_t = c_w / c_t, the decaying roots
// p = sqrt(n^2 - cos^2 g), s = sqrt(n_t^2 - cos^2 g), and C = cos 2theta_t = 2 cos^2 g / n_t^2 - 1,
//     R = (m sin g (C^2 s + (1 - C^2) p) - p s) / (m sin g (C^2 s + (1 - C^2) p) + p s).
// Attenuations are in dB per wavelength, entering as imaginary parts of the speeds (the conversion is ours).
// A fluid (shear speed 0) is not this function's: each caller keeps its own fluid expression, unchanged.
// No CUDA twin: no kernel evaluates a bottom reflection (the coefficient is per ray or per image, on the host).

#include <cmath>
#include <complex>

namespace sonar {

/// Complex quotient by Smith's method with a reciprocal, the operation order NumPy uses, so that amplitudes agree
/// with the Python models (python/gpusonar/sim/propagation.py) to the last bits.
inline std::complex<double> smith_divide(std::complex<double> a, std::complex<double> b) {
  if (std::abs(b.real()) >= std::abs(b.imag())) {
    const double rat = b.imag() / b.real();
    const double scl = 1.0 / (b.real() + b.imag() * rat);
    return {(a.real() + a.imag() * rat) * scl, (a.imag() - a.real() * rat) * scl};
  }
  const double rat = b.real() / b.imag();
  const double scl = 1.0 / (b.imag() + b.real() * rat);
  return {(a.real() * rat + a.imag()) * scl, (a.imag() * rat - a.real()) * scl};
}

/// R of an elastic half-space at a grazing angle (rad). `speed` m/s, `density_ratio` rho_b / rho_w, `attenuation`
/// dB per wavelength of the compressional wave; `shear_speed` m/s (> 0), `shear_attenuation` dB per wavelength.
inline std::complex<double> elastic_reflection(double grazing, double c_water, double speed, double density_ratio, double attenuation, double shear_speed,
                                               double shear_attenuation) {
  using cd = std::complex<double>;
  constexpr double kPi = 3.14159265358979323846;
  const double delta = attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));  // loss tangent
  const cd c_b = speed / cd{1.0, delta};
  const cd n = c_water / c_b;
  const double cg = std::cos(grazing);
  cd root = std::sqrt(n * n - cg * cg + cd{0.0, 0.0});
  if (root.imag() < 0.0) root = -root;  // decaying branch
  const double ms = density_ratio * std::sin(grazing);
  const double delta_t = shear_attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
  const cd n_t = c_water * cd{1.0, delta_t} / shear_speed;
  cd s = std::sqrt(n_t * n_t - cg * cg + cd{0.0, 0.0});
  if (s.imag() < 0.0) s = -s;
  const cd cos2t = 2.0 * cg * cg / (n_t * n_t) - 1.0;
  const cd num = ms * (cos2t * cos2t * s + (1.0 - cos2t * cos2t) * root);
  return smith_divide(num - root * s, num + root * s);
}

}  // namespace sonar
