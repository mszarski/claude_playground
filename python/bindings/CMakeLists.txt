pybind11_add_module(_gpusonar module.cpp bind_sim.cpp bind_recon.cpp bind_detect.cpp)
target_link_libraries(_gpusonar PRIVATE sonar::core sonar::io sonar::sim sonar::dsp sonar::recon sonar::detect)
# Build the extension straight into the Python package so `PYTHONPATH=python` works.
set_target_properties(_gpusonar PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/gpusonar")

target_compile_definitions(_gpusonar PRIVATE SONAR_HAVE_MOTION=$<BOOL:${SONAR_HAVE_MOTION}>)
if(SONAR_HAVE_MOTION)
  target_sources(_gpusonar PRIVATE bind_motion.cpp)
  target_link_libraries(_gpusonar PRIVATE sonar::motion)
endif()
