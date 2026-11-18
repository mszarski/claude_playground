// Kernels of the far-field beamformers in their batched form (beamform_batched.hpp), written
// once for the host and for CUDA. Each is a plain map: element i writes its own outputs.
//
// Complex arrays are interleaved (re, im). Samples and weights applied to samples are single
// precision, as in the reference host code (beamform.cpp); covariances, their Cholesky
// factors and the solves are double precision, as there.
//
//   FarfieldSumKernel    one (bearing, frequency bin): the weighted, steered sum over elements
//   MvdrBandKernel       one (sub-band, channel, bin): the channel's spectrum inside the sub-band
//   MvdrTransposeKernel  [sub-band][channel][time] -> [sub-band][time][channel]
//   MvdrOuterKernel      one element of the lower triangle of one range block's x x^H
//   MvdrTotalKernel      the same summed over the blocks
//   MvdrFactorKernel     one (sub-band, block): training covariance, loading, Cholesky in place
//   MvdrSteerKernel      one (sub-band, bearing, element): the steering vector's entry
//   MvdrSolveKernel      one (sub-band, block, bearing): w = R^-1 a / (a^H R^-1 a)
//   MvdrApplyKernel      one (group of bearings, time sample): the weights applied, sub-bands summed
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::live {

/// Bearings whose weights one element of MvdrApplyKernel applies to its time sample.
constexpr std::int64_t kMvdrApplyGroup = 8;

/// cos and sin of 2 pi `turns`; the turns are reduced in double precision first (a steering
/// phase at the carrier is hundreds of radians).
SONAR_HD inline void beam_phasor(double turns, float& c, float& s) {
  const double frac = turns - std::floor(turns);
  const float a = static_cast<float>(6.283185307179586476925 * frac);
  c = std::cos(a);
  s = std::sin(a);
}

/// Frequency of FFT bin k as a signed multiple of the bin spacing (numpy.fft.fftfreq order).
SONAR_HD inline std::int64_t beam_signed_bin(std::int64_t k, std::int64_t n) { return k < (n + 1) / 2 ? k : k - n; }

struct FarfieldSumKernel {
  const float* x;     ///< [n_el][n] complex: spectrum of every channel
  const float* w;     ///< [n_el] aperture weights
  const double* tau;  ///< [n_b][n_el] plane-wave leads, s
  float* out;         ///< [n_b][n] complex: spectrum of every beam, times `scale`
  std::int64_t n, n_el;
  double df, fc;
  float scale;        ///< 1 / n: the inverse transform that follows is not scaled

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / n, k = i % n;
    const double f = fc + static_cast<double>(beam_signed_bin(k, n)) * df;
    const double* t = tau + b * n_el;
    float re = 0.0F, im = 0.0F;
    for (std::int64_t e = 0; e < n_el; ++e) {
      float c, s;
      beam_phasor(-(f * t[e]), c, s);
      const float xr = w[e] * x[2 * (e * n + k)], xi = w[e] * x[2 * (e * n + k) + 1];
      re += xr * c - xi * s;
      im += xr * s + xi * c;
    }
    out[2 * i] = re * scale;
    out[2 * i + 1] = im * scale;
  }
};

struct MvdrBandKernel {
  const float* x;            ///< [n_el][n] complex spectra
  const std::int32_t* band;  ///< [n] sub-band of each bin, or -1
  const std::int32_t* sub;   ///< [n_sub] the sub-bands of this run
  float* y;                  ///< [n_sub][n_el][n] complex
  std::int64_t n, n_el;
  float scale;               ///< 1 / n

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n, c = (i / n) % n_el, s = i / (n * n_el);
    if (band[k] == sub[s]) {
      y[2 * i] = x[2 * (c * n + k)] * scale;
      y[2 * i + 1] = x[2 * (c * n + k) + 1] * scale;
    } else {
      y[2 * i] = 0.0F;
      y[2 * i + 1] = 0.0F;
    }
  }
};

struct MvdrTransposeKernel {
  const float* y;  ///< [n_sub][n_el][n] complex
  float* xt;       ///< [n_sub][n][n_el] complex
  std::int64_t n, n_el;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t e = i % n_el, t = (i / n_el) % n, s = i / (n * n_el);
    const std::int64_t src = (s * n_el + e) * n + t;
    xt[2 * i] = y[2 * src];
    xt[2 * i + 1] = y[2 * src + 1];
  }
};

struct MvdrOuterKernel {
  const float* xt;  ///< [n_sub][n][n_el] complex
  double* outer;    ///< [n_sub][n_blocks][n_el][n_el] complex; the lower triangle is written
  std::int64_t n, n_el, n_blocks, block;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t q = index % n_el, p = (index / n_el) % n_el;
    if (q > p) return;
    const std::int64_t i = (index / (n_el * n_el)) % n_blocks, s = index / (n_el * n_el * n_blocks);
    const std::int64_t t0 = i * block, t1 = t0 + block < n ? t0 + block : n;
    const float* row = xt + 2 * (s * n + t0) * n_el;
    double re = 0.0, im = 0.0;
    for (std::int64_t t = t0; t < t1; ++t, row += 2 * n_el) {
      const double ar = row[2 * p], ai = row[2 * p + 1], br = row[2 * q], bi = row[2 * q + 1];
      re += ar * br + ai * bi;   // x_p conj(x_q)
      im += ai * br - ar * bi;
    }
    outer[2 * index] = re;
    outer[2 * index + 1] = im;
  }
};

struct MvdrTotalKernel {
  const double* outer;  ///< [n_sub][n_blocks][n_el][n_el] complex
  double* total;        ///< [n_sub][n_el][n_el] complex; the lower triangle is written
  std::int64_t n_el, n_blocks;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t q = index % n_el, p = (index / n_el) % n_el;
    if (q > p) return;
    const std::int64_t s = index / (n_el * n_el), at = p * n_el + q;
    double re = 0.0, im = 0.0;
    for (std::int64_t i = 0; i < n_blocks; ++i) {
      const double* m = outer + 2 * ((s * n_blocks + i) * n_el * n_el + at);
      re += m[0];
      im += m[1];
    }
    total[2 * index] = re;
    total[2 * index + 1] = im;
  }
};

struct MvdrFactorKernel {
  const double* outer;   ///< [n_sub][n_blocks][n_el][n_el] complex
  const double* total;   ///< [n_sub][n_el][n_el] complex
  double* r;             ///< [n_sub][n_blocks][n_el][n_el] complex: the Cholesky factor (lower) out
  std::int32_t* status;  ///< [n_sub][n_blocks]: 0, or 1 when the covariance is not positive definite
  std::int64_t n, n_el, n_blocks, block, guard_blocks;
  double loading;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t i = index % n_blocks, s = index / n_blocks;
    const std::int64_t nn = n_el * n_el;
    // Training data: every block but this one and its guards.
    const std::int64_t j0 = i - guard_blocks > 0 ? i - guard_blocks : 0;
    const std::int64_t j1 = i + guard_blocks + 1 < n_blocks ? i + guard_blocks + 1 : n_blocks;
    std::int64_t count = n;
    for (std::int64_t j = j0; j < j1; ++j) count -= (j + 1) * block <= n ? block : n - j * block;
    const double* tot = total + 2 * s * nn;
    const double* o = outer + 2 * s * n_blocks * nn;
    double* m = r + 2 * index * nn;
    const double inv = 1.0 / static_cast<double>(count);
    double trace = 0.0;
    for (std::int64_t p = 0; p < n_el; ++p) {
      for (std::int64_t q = 0; q <= p; ++q) {
        const std::int64_t at = 2 * (p * n_el + q);
        double re = tot[at], im = tot[at + 1];
        for (std::int64_t j = j0; j < j1; ++j) {
          re -= o[2 * j * nn + at];
          im -= o[2 * j * nn + at + 1];
        }
        m[at] = re * inv;
        m[at + 1] = p == q ? 0.0 : im * inv;
      }
      trace += m[2 * (p * n_el + p)];
    }
    const double load = loading * (trace / static_cast<double>(n_el));
    for (std::int64_t p = 0; p < n_el; ++p) m[2 * (p * n_el + p)] += load;
    // Cholesky, lower triangle in place (the diagonal is real).
    for (std::int64_t j = 0; j < n_el; ++j) {
      double d = m[2 * (j * n_el + j)];
      for (std::int64_t k = 0; k < j; ++k) {
        const double lr = m[2 * (j * n_el + k)], li = m[2 * (j * n_el + k) + 1];
        d -= lr * lr + li * li;
      }
      if (!(d > 0.0) || !(d < 1e300)) {
        status[index] = 1;
        return;
      }
      const double diag = std::sqrt(d);
      m[2 * (j * n_el + j)] = diag;
      m[2 * (j * n_el + j) + 1] = 0.0;
      for (std::int64_t p = j + 1; p < n_el; ++p) {
        double re = m[2 * (p * n_el + j)], im = m[2 * (p * n_el + j) + 1];
        for (std::int64_t k = 0; k < j; ++k) {
          const double ar = m[2 * (p * n_el + k)], ai = m[2 * (p * n_el + k) + 1];
          const double br = m[2 * (j * n_el + k)], bi = m[2 * (j * n_el + k) + 1];
          re -= ar * br + ai * bi;   // L_pk conj(L_jk)
          im -= ai * br - ar * bi;
        }
        m[2 * (p * n_el + j)] = re / diag;
        m[2 * (p * n_el + j) + 1] = im / diag;
      }
    }
    status[index] = 0;
  }
};

/// The steering vectors of each sub-band, a[s][b][e] = exp(2 pi i f_s tau[b][e]), double
/// precision: every range block of the sub-band solves against the same ones.
struct MvdrSteerKernel {
  const double* tau;    ///< [n_b][n_el] plane-wave leads, s
  const double* f_abs;  ///< [n_sub] centre frequency of each sub-band, Hz
  double* a;            ///< [n_sub][n_b][n_el] complex
  std::int64_t n_el, n_b;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t s = index / (n_b * n_el);
    const double turns = f_abs[s] * tau[index % (n_b * n_el)];
    const double angle = 6.283185307179586476925 * (turns - std::floor(turns));
    a[2 * index] = std::cos(angle);
    a[2 * index + 1] = std::sin(angle);
  }
};

struct MvdrSolveKernel {
  const double* r;             ///< [n_sub][n_blocks][n_el][n_el] complex: Cholesky factors (lower)
  const std::int32_t* status;  ///< [n_sub][n_blocks]
  const double* a;             ///< [n_sub][n_b][n_el] complex steering vectors
  double* work;                ///< [n_sub][n_blocks][n_b][n_el] complex: R^-1 a
  float* coeff;                ///< [n_sub][n_blocks][n_b][n_el] complex: n_el conj(w)
  std::int32_t* singular;      ///< [n_sub][n_blocks][n_b]: 1 where a^H R^-1 a is zero or not finite
  std::int64_t n_el, n_blocks, n_b;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t b = index % n_b, si = index / n_b, s = si / n_blocks;
    float* cf = coeff + 2 * index * n_el;
    if (status[si] != 0) {
      for (std::int64_t e = 0; e < 2 * n_el; ++e) cf[e] = 0.0F;
      singular[index] = 0;
      return;
    }
    const double* l = r + 2 * si * n_el * n_el;
    const double* av = a + 2 * (s * n_b + b) * n_el;
    double* z = work + 2 * index * n_el;
    // Forward: L y = a.
    for (std::int64_t e = 0; e < n_el; ++e) {
      double re = av[2 * e], im = av[2 * e + 1];
      for (std::int64_t j = 0; j < e; ++j) {
        const double lr = l[2 * (e * n_el + j)], li = l[2 * (e * n_el + j) + 1];
        re -= lr * z[2 * j] - li * z[2 * j + 1];
        im -= lr * z[2 * j + 1] + li * z[2 * j];
      }
      const double d = l[2 * (e * n_el + e)];
      z[2 * e] = re / d;
      z[2 * e + 1] = im / d;
    }
    // Backward: L^H z = y.
    for (std::int64_t e = n_el - 1; e >= 0; --e) {
      double re = z[2 * e], im = z[2 * e + 1];
      for (std::int64_t j = e + 1; j < n_el; ++j) {
        const double lr = l[2 * (j * n_el + e)], li = -l[2 * (j * n_el + e) + 1];   // conj(L_je)
        re -= lr * z[2 * j] - li * z[2 * j + 1];
        im -= lr * z[2 * j + 1] + li * z[2 * j];
      }
      const double d = l[2 * (e * n_el + e)];
      z[2 * e] = re / d;
      z[2 * e + 1] = im / d;
    }
    // gain = a^H R^-1 a.
    double gr = 0.0, gi = 0.0;
    for (std::int64_t e = 0; e < n_el; ++e) {
      const double ar = av[2 * e], ai = av[2 * e + 1];
      gr += ar * z[2 * e] + ai * z[2 * e + 1];
      gi += ar * z[2 * e + 1] - ai * z[2 * e];
    }
    const double g2 = gr * gr + gi * gi;
    if (!(g2 > 0.0) || !(g2 < 1e300)) {
      for (std::int64_t e = 0; e < 2 * n_el; ++e) cf[e] = 0.0F;
      singular[index] = 1;
      return;
    }
    // n_el conj(z / gain).
    const double scale = static_cast<double>(n_el) / g2;
    for (std::int64_t e = 0; e < n_el; ++e) {
      const double qr = z[2 * e] * gr + z[2 * e + 1] * gi;   // z conj(gain)
      const double qi = z[2 * e + 1] * gr - z[2 * e] * gi;
      cf[2 * e] = static_cast<float>(scale * qr);
      cf[2 * e + 1] = static_cast<float>(-(scale * qi));
    }
    singular[index] = 0;
  }
};

struct MvdrApplyKernel {
  const float* xt;     ///< [n_sub][n][n_el] complex
  const float* coeff;  ///< [n_sub][n_blocks][n_b][n_el] complex
  float* out;          ///< [n_b][n] complex; added to
  std::int64_t n, n_el, n_blocks, n_b, n_sub, block;

  SONAR_HD void operator()(std::int64_t index) const {
    const std::int64_t t = index % n, b0 = (index / n) * kMvdrApplyGroup;
    const std::int64_t rows = b0 + kMvdrApplyGroup <= n_b ? kMvdrApplyGroup : n_b - b0;
    const std::int64_t i = t / block;
    float re[kMvdrApplyGroup], im[kMvdrApplyGroup];
    for (std::int64_t g = 0; g < kMvdrApplyGroup; ++g) {
      re[g] = 0.0F;
      im[g] = 0.0F;
    }
    for (std::int64_t s = 0; s < n_sub; ++s) {
      const float* x = xt + 2 * (s * n + t) * n_el;
      const float* c = coeff + 2 * ((s * n_blocks + i) * n_b + b0) * n_el;
      for (std::int64_t g = 0; g < rows; ++g, c += 2 * n_el) {
        float ar = 0.0F, ai = 0.0F;
        for (std::int64_t e = 0; e < n_el; ++e) {
          ar += c[2 * e] * x[2 * e] - c[2 * e + 1] * x[2 * e + 1];
          ai += c[2 * e] * x[2 * e + 1] + c[2 * e + 1] * x[2 * e];
        }
        re[g] += ar;
        im[g] += ai;
      }
    }
    for (std::int64_t g = 0; g < rows; ++g) {
      out[2 * ((b0 + g) * n + t)] += re[g];
      out[2 * ((b0 + g) * n + t) + 1] += im[g];
    }
  }
};

}  // namespace sonar::live
