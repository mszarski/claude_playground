// The per-iteration work of SBL as three kernels, written once for the host and for CUDA.
//
// Given the source powers gamma on the bearings still in play and the noise sigma2 per
// frequency, each frequency f has the model covariance
//     S_f = sigma2_f I + sum_m gamma_m a_fm a_fm^H            (N x N, Hermitian)
// and the update needs, for every bearing m in play,
//     den_fm = a_fm^H S_f^-1 a_fm        num_fm = (1/L) sum_l |a_fm^H S_f^-1 y_fl|^2.
// With S_f = C C^H (Cholesky), w = C^-1 a and z = C^-1 y, these are |w|^2 and |w^H z|^2:
// one triangular solve per bearing instead of a full solve.
//
//   SblCovKernel       one element of the lower triangle of one S_f
//   SblFactorKernel    one frequency: Cholesky in place, then z_l = C^-1 y_l
//   SblUpdateKernel    one (frequency, bearing): w = C^-1 a, then den and num
//
// Complex arrays are interleaved (re, im) doubles; std::complex<double> has that layout.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::live {

/// The update kernel keeps one solved steering vector on the stack.
constexpr std::int64_t kSblMaxChannels = 64;

struct SblCovKernel {
  const double* a;             ///< [F][M][N] complex: steering vector of bearing m at frequency f
  const double* gamma;         ///< [M]
  const std::int32_t* active;  ///< [n_active] bearings in play
  const double* sigma2;        ///< [F]
  double* cov;                 ///< [F][N][N] complex, row major; the lower triangle is written
  std::int64_t n, m, n_active;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t f = index / (n * n), rest = index % (n * n), i = rest / n, j = rest % n;
    if (j > i) return;
    const double* af = a + 2 * f * m * n;
    double re = i == j ? sigma2[f] : 0.0, im = 0.0;
    for (std::int64_t k = 0; k < n_active; ++k) {
      const std::int64_t col = active[k];
      const double* v = af + 2 * col * n;
      const double g = gamma[col];
      const double ar = v[2 * i], ai = v[2 * i + 1], br = v[2 * j], bi = v[2 * j + 1];
      re += g * (ar * br + ai * bi);          // a_i conj(a_j)
      im += g * (ai * br - ar * bi);
    }
    double* out = cov + 2 * (f * n * n + i * n + j);
    out[0] = re;
    out[1] = im;
  }
};

struct SblFactorKernel {
  double* cov;      ///< [F][N][N] complex: S_f in, its Cholesky factor C (lower) out
  const double* y;  ///< [F][L][N] complex snapshots
  double* z;        ///< [F][L][N] complex: C^-1 y
  std::int64_t n, n_snapshots;

  SONAR_HD void operator()(std::int64_t f) const {
    double* c = cov + 2 * f * n * n;
    for (std::int64_t j = 0; j < n; ++j) {
      double d = c[2 * (j * n + j)];
      for (std::int64_t k = 0; k < j; ++k) {
        const double* e = c + 2 * (j * n + k);
        d -= e[0] * e[0] + e[1] * e[1];
      }
      // S_f is positive definite by construction (sigma2 > 0); the floor only guards rounding.
      const double diag = std::sqrt(d > 1e-300 ? d : 1e-300);
      c[2 * (j * n + j)] = diag;
      c[2 * (j * n + j) + 1] = 0.0;
      for (std::int64_t i = j + 1; i < n; ++i) {
        double re = c[2 * (i * n + j)], im = c[2 * (i * n + j) + 1];
        for (std::int64_t k = 0; k < j; ++k) {
          const double* p = c + 2 * (i * n + k);
          const double* q = c + 2 * (j * n + k);
          re -= p[0] * q[0] + p[1] * q[1];    // C_ik conj(C_jk)
          im -= p[1] * q[0] - p[0] * q[1];
        }
        c[2 * (i * n + j)] = re / diag;
        c[2 * (i * n + j) + 1] = im / diag;
      }
    }
    for (std::int64_t l = 0; l < n_snapshots; ++l) {
      const double* yl = y + 2 * (f * n_snapshots + l) * n;
      double* zl = z + 2 * (f * n_snapshots + l) * n;
      for (std::int64_t i = 0; i < n; ++i) {
        double re = yl[2 * i], im = yl[2 * i + 1];
        for (std::int64_t k = 0; k < i; ++k) {
          const double* p = c + 2 * (i * n + k);
          re -= p[0] * zl[2 * k] - p[1] * zl[2 * k + 1];
          im -= p[0] * zl[2 * k + 1] + p[1] * zl[2 * k];
        }
        const double diag = c[2 * (i * n + i)];
        zl[2 * i] = re / diag;
        zl[2 * i + 1] = im / diag;
      }
    }
  }
};

struct SblUpdateKernel {
  const double* a;             ///< [F][M][N] complex
  const double* cov;           ///< [F][N][N] complex: the Cholesky factors
  const double* z;             ///< [F][L][N] complex
  const std::int32_t* active;  ///< [n_active]
  double* num;                 ///< [F][n_active]
  double* den;                 ///< [F][n_active]
  std::int64_t n, m, n_active, n_snapshots;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t f = index / n_active, k = index % n_active;
    const double* v = a + 2 * (f * m + active[k]) * n;
    const double* c = cov + 2 * f * n * n;
    double w[2 * kSblMaxChannels];
    double norm = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      double re = v[2 * i], im = v[2 * i + 1];
      for (std::int64_t q = 0; q < i; ++q) {
        const double* p = c + 2 * (i * n + q);
        re -= p[0] * w[2 * q] - p[1] * w[2 * q + 1];
        im -= p[0] * w[2 * q + 1] + p[1] * w[2 * q];
      }
      const double diag = c[2 * (i * n + i)];
      w[2 * i] = re / diag;
      w[2 * i + 1] = im / diag;
      norm += w[2 * i] * w[2 * i] + w[2 * i + 1] * w[2 * i + 1];
    }
    double sum = 0.0;
    for (std::int64_t l = 0; l < n_snapshots; ++l) {
      const double* zl = z + 2 * (f * n_snapshots + l) * n;
      double re = 0.0, im = 0.0;
      for (std::int64_t i = 0; i < n; ++i) {
        re += w[2 * i] * zl[2 * i] + w[2 * i + 1] * zl[2 * i + 1];      // conj(w_i) z_i
        im += w[2 * i] * zl[2 * i + 1] - w[2 * i + 1] * zl[2 * i];
      }
      sum += re * re + im * im;
    }
    num[index] = sum / static_cast<double>(n_snapshots);
    den[index] = norm;
  }
};

}  // namespace sonar::live
