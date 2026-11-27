set(_live_sources src/cluster.cpp src/tracker.cpp src/classify.cpp src/doa.cpp src/chain.cpp src/udp_source.cpp src/board_merge.cpp src/sync.cpp src/options.cpp src/beamform.cpp src/calibrate.cpp src/elevation.cpp src/elevation_multipath.cpp src/environment.cpp src/decimate.cpp src/tonal.cpp src/derive.cpp src/detector.cpp src/multipath_depth.cpp src/multipath_rays.cpp src/coverage.cpp src/raytrace.cpp src/doppler.cpp src/doppler_batched.cpp)
if(SONAR_HAVE_CUDA)
  list(APPEND _live_sources src/doa_cuda.cu src/beamform_cuda.cu src/detector_cuda.cu src/doppler_cuda.cu)
  # The detector's statistic must give the host's flags: no product contracted into a fused multiply-add (detector_kernel.hpp).
  # The Doppler stages' folds and background likewise (doppler_batched.hpp).
  set_source_files_properties(src/detector_cuda.cu src/doppler_cuda.cu PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
endif()
# The board merge's alignment must round as the Python engine's does (board_merge.py): no fused multiply-add.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/board_merge.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
add_library(sonar_live ${_live_sources})
add_library(sonar::live ALIAS sonar_live)
target_include_directories(sonar_live PUBLIC include)
target_link_libraries(sonar_live PUBLIC sonar::core sonar::dsp sonar::recon sonar::detect Eigen3::Eigen PRIVATE sonar_warnings)
