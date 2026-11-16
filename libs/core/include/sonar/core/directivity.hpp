// Element directivity shared by the simulator (forward model) and the
// backprojection kernel (weighting and ray culling), so both agree.
//
// Model: a rectangular aperture, separable in the two direction cosines.
//   D = sinc(k_az * sin_az) * sinc(k_el * sin_el),  k = 0.886 / beamwidth
// which is -3 dB (one way, power) at half the FWHM beamwidth. Beamwidth scales
// inversely with frequency through `freq_ratio = f / f_ref`. NaN beamwidth means
// omnidirectional in that plane. A directional element radiates into its front
// hemisphere only.
#pragma once

#include <cmath>

#include "sonar/core/vec.hpp"

namespace sonar {

/// Axes that split a direction into azimuth and elevation for an element.
/// The elevation axis is the part of body-down orthogonal to the boresight
/// (body-forward if the element looks straight down or up); azimuth completes it.
/// Side-looking element: azimuth is along-track. Forward-looking: athwartships.
template <typename T>
struct ElementAxes {
  Vec3<T> boresight, az, el;
};

template <typename T>
SONAR_INLINE ElementAxes<T> element_axes(const Vec3<T>& boresight) {
  ElementAxes<T> a;
  a.boresight = normalized(boresight);
  Vec3<T> ref{T(0), T(0), T(1)};
  T along = dot(ref, a.boresight);
  if (std::abs(along) > T(0.999)) {
    ref = Vec3<T>{T(1), T(0), T(0)};
    along = dot(ref, a.boresight);
    a.az = normalized(ref - a.boresight * along);
    a.el = cross(a.boresight, a.az);
    return a;
  }
  a.el = normalized(ref - a.boresight * along);
  a.az = cross(a.el, a.boresight);
  return a;
}

template <typename T>
SONAR_INLINE T sinc_pi(T x) {
  // sin(pi x) / (pi x)
  const T pi = T(3.14159265358979323846);
  const T a = pi * x;
  if (std::abs(a) < T(1e-4)) return T(1) - a * a / T(6);
  return std::sin(a) / a;
}

/// Amplitude response towards unit direction `dir` (same frame as the axes).
template <typename T>
SONAR_INLINE T directivity(const ElementAxes<T>& axes, T beamwidth_az, T beamwidth_el,
                           const Vec3<T>& dir, T freq_ratio = T(1)) {
  const bool omni_az = !(beamwidth_az == beamwidth_az);  // NaN test that survives -ffast-math-free builds
  const bool omni_el = !(beamwidth_el == beamwidth_el);
  if (omni_az && omni_el) return T(1);
  if (dot(dir, axes.boresight) <= T(0)) return T(0);
  T d = T(1);
  if (!omni_az) d *= sinc_pi(T(0.886) * freq_ratio * dot(dir, axes.az) / beamwidth_az);
  if (!omni_el) d *= sinc_pi(T(0.886) * freq_ratio * dot(dir, axes.el) / beamwidth_el);
  return d;
}

}  // namespace sonar
