// CPU FFT wrappers (pocketfft). Forward is unnormalised; inverse scales by 1/n.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace sonar::dsp {

void fft(std::complex<float>* data, std::size_t n, bool inverse);
void fft(std::vector<std::complex<float>>& data, bool inverse);

/// Smallest length >= n whose prime factors are all <= 11 (fast for the FFT).
std::size_t good_fft_size(std::size_t n);

}  // namespace sonar::dsp
