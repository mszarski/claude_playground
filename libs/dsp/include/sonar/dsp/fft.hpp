// CPU FFT wrappers (pocketfft). Forward is unnormalised; inverse scales by 1/n.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace sonar::dsp {

void fft(std::complex<float>* data, std::size_t n, bool inverse);
void fft(std::vector<std::complex<float>>& data, bool inverse);

/// The same transform without the inverse's 1/n (what cuFFT computes): for code that runs a
/// device formulation on the host (sonar/dsp/exec_host.hpp).
void fft_unscaled(std::complex<float>* data, std::size_t n, bool inverse);

/// Smallest length >= n whose prime factors are all <= 11 (fast for the FFT).
std::size_t good_fft_size(std::size_t n);

}  // namespace sonar::dsp
