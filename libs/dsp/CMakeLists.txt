add_library(sonar_dsp
  src/fft.cpp
  src/baseband.cpp
  src/conditioning.cpp
  src/pulse_compression.cpp)
add_library(sonar::dsp ALIAS sonar_dsp)
target_include_directories(sonar_dsp PUBLIC include)
# Vendored header: keep its warnings out of our build.
target_include_directories(sonar_dsp SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/pocketfft)
target_link_libraries(sonar_dsp PUBLIC sonar::core PRIVATE sonar_warnings)
