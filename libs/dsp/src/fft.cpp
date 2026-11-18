#include "sonar/dsp/fft.hpp"

#include "pocketfft_hdronly.h"

namespace sonar::dsp {

void fft(std::complex<float>* data, std::size_t n, bool inverse) {
  if (n == 0) return;
  const pocketfft::shape_t shape{n};
  const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
  const float scale = inverse ? 1.0F / static_cast<float>(n) : 1.0F;
  pocketfft::c2c(shape, stride, stride, {0}, !inverse, data, data, scale);
}

void fft_unscaled(std::complex<float>* data, std::size_t n, bool inverse) {
  if (n == 0) return;
  const pocketfft::shape_t shape{n};
  const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
  pocketfft::c2c(shape, stride, stride, {0}, !inverse, data, data, 1.0F);
}

void fft(std::vector<std::complex<float>>& data, bool inverse) { fft(data.data(), data.size(), inverse); }

std::size_t good_fft_size(std::size_t n) { return pocketfft::detail::util::good_size_cmplx(n); }

}  // namespace sonar::dsp
