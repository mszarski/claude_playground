// Passive tonal analysis (libs/live tonal.hpp): the analyzer, the Goertzel sum and the
// harmonic grouping, for gpusonar/tonal.py to be held against.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/tonal.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

void bind_tonal(py::module_& m) {
  py::class_<live::TonalAnalyzer>(m, "TonalAnalyzerCpp")
      .def(py::init([](const py::dict& options) {
             live::TonalOptions o;
             double f_low = 0.0, f_high = 0.0;
             for (const auto& item : options) {
               const auto key = item.first.cast<std::string>();
               const std::string value = py::str(item.second);
               if (key == "f_low") f_low = std::atof(value.c_str());
               else if (key == "f_high") f_high = std::atof(value.c_str());
               else if (!live::set_tonal_option(o, key, value)) throw std::invalid_argument("unknown option " + key);
             }
             return live::TonalAnalyzer(o, f_low, f_high);
           }),
           "Options as gpusonar.tonal.cpp_options makes them ({\"tonal_goertzel_lines\": \"300,450\", ...}).")
      .def("push",
           [](live::TonalAnalyzer& a, const PingRecord& record) {
             std::string json;
             {
               py::gil_scoped_release release;
               json = live::tonal_events_json(a.push(record), a);
             }
             return json;
           },
           "The results completed by this record's samples, as a JSON array (the events of gpusonar.tonal.TonalAnalyzer.push).")
      .def("gram", [](const live::TonalAnalyzer& a) {
        py::array_t<float> db({static_cast<py::ssize_t>(a.gram_frequencies.empty() ? 0 : a.bearings.size()),
                               static_cast<py::ssize_t>(a.gram_frequencies.size())});
        std::copy(a.gram_db.begin(), a.gram_db.end(), db.mutable_data());
        return py::make_tuple(a.bearings, a.gram_frequencies, db);
      }, "The latest DEMONgram: (bearings rad, modulation frequencies Hz, dB above the background [bearing, frequency]).");

  m.def(
      "tonal_goertzel",
      [](const py::array& x, double frequency, double sample_rate) {
        if (x.ndim() != 1) throw std::invalid_argument("tonal_goertzel: one channel of samples");
        if (x.dtype().kind() == 'c') {
          const auto a = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>::ensure(x);
          return live::goertzel(a.data(), static_cast<std::int64_t>(a.size()), frequency, sample_rate);
        }
        const auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(x);
        return live::goertzel(a.data(), static_cast<std::int64_t>(a.size()), frequency, sample_rate);
      },
      py::arg("x"), py::arg("frequency"), py::arg("sample_rate"),
      "sum_n x[n] exp(-2 pi j f n / fs) by the Goertzel recurrence (single-precision samples).");
  m.def(
      "tonal_harmonic_set",
      [](const std::vector<double>& frequencies, const std::vector<double>& weights, double df, const py::dict& options) {
        live::TonalOptions o;
        for (const auto& item : options) {
          live::set_tonal_option(o, item.first.cast<std::string>(), py::str(item.second));
        }
        live::DemonSource src;
        for (std::size_t i = 0; i < frequencies.size(); ++i) {
          live::DemonLine l;
          l.frequency = frequencies[i];
          l.snr_db = weights[i];
          src.lines.push_back(l);
        }
        live::harmonic_set(src, df, o.demon);
        std::vector<std::int64_t> h;
        for (const auto& l : src.lines) h.push_back(l.harmonic);
        return py::make_tuple(src.shaft_hz, src.blade_hz, src.blades, src.confidence, h);
      },
      py::arg("frequencies"), py::arg("weights"), py::arg("df"), py::arg("options"),
      "(shaft Hz, blade Hz, blades, confidence, harmonic of each line) from modulation lines; 0 where not known.");
  m.def(
      "tonal_demon_threshold",
      [](double pfa, std::int64_t n_spectra, double correlation, std::int64_t background) {
        double n = 0.0;
        const double t = live::demon_threshold(pfa, n_spectra, correlation, background, &n);
        return py::make_tuple(t, n);
      },
      py::arg("pfa"), py::arg("n_spectra"), py::arg("correlation"), py::arg("background") = 0,
      "(threshold as a ratio to the background, independent spectra) of the DEMON line detector.");
}
