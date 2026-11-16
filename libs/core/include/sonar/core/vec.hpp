// Small fixed-size vector and matrix types usable in host and device code.
// Frame convention everywhere: north-east-down (NED); body axes x forward,
// y starboard, z down. Angles in radians.
#pragma once

#include <cmath>

#include "sonar/core/platform.hpp"

namespace sonar {

template <typename T>
struct Vec3 {
  T x{}, y{}, z{};

  SONAR_HD constexpr Vec3() = default;
  SONAR_HD constexpr Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
  template <typename U>
  SONAR_HD explicit constexpr Vec3(const Vec3<U>& o)
      : x(static_cast<T>(o.x)), y(static_cast<T>(o.y)), z(static_cast<T>(o.z)) {}

  SONAR_HD constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  SONAR_HD constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  SONAR_HD constexpr Vec3 operator*(T s) const { return {x * s, y * s, z * s}; }
  SONAR_HD constexpr Vec3 operator/(T s) const { return {x / s, y / s, z / s}; }
  SONAR_HD constexpr Vec3 operator-() const { return {-x, -y, -z}; }
  SONAR_HD constexpr Vec3& operator+=(const Vec3& o) {
    x += o.x; y += o.y; z += o.z;
    return *this;
  }
  SONAR_HD constexpr T operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

template <typename T>
SONAR_INLINE constexpr T dot(const Vec3<T>& a, const Vec3<T>& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
SONAR_INLINE constexpr Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <typename T>
SONAR_INLINE T norm(const Vec3<T>& a) {
  return std::sqrt(dot(a, a));
}
template <typename T>
SONAR_INLINE Vec3<T> normalized(const Vec3<T>& a) {
  return a / norm(a);
}

using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;

/// Row-major 3x3 matrix.
template <typename T>
struct Mat3 {
  T m[3][3]{};

  SONAR_HD constexpr Vec3<T> operator*(const Vec3<T>& v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
  }
  SONAR_HD constexpr Mat3 operator*(const Mat3& o) const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
    return r;
  }
  SONAR_HD constexpr Mat3 transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
    return r;
  }
};

using Mat3d = Mat3<double>;

/// Body-to-NED rotation, ASASIN Eq. 2: R = Rz(yaw) Ry(pitch) Rx(roll).
/// Positive roll lowers starboard, positive pitch raises the bow, positive yaw
/// rotates to starboard.
template <typename T>
SONAR_INLINE Mat3<T> rotation_ned(T roll, T pitch, T yaw) {
  const T cr = std::cos(roll), sr = std::sin(roll);
  const T cp = std::cos(pitch), sp = std::sin(pitch);
  const T cy = std::cos(yaw), sy = std::sin(yaw);
  Mat3<T> r;
  r.m[0][0] = cp * cy;
  r.m[0][1] = cy * sp * sr - cr * sy;
  r.m[0][2] = cr * cy * sp + sr * sy;
  r.m[1][0] = cp * sy;
  r.m[1][1] = cr * cy + sp * sr * sy;
  r.m[1][2] = cr * sp * sy - cy * sr;
  r.m[2][0] = -sp;
  r.m[2][1] = cp * sr;
  r.m[2][2] = cp * cr;
  return r;
}

}  // namespace sonar
