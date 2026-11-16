pybind11_add_module(_gpusonar module.cpp bind_sim.cpp bind_recon.cpp)
target_link_libraries(_gpusonar PRIVATE sonar::core sonar::io sonar::sim sonar::dsp sonar::recon)
# Build the extension straight into the Python package so `PYTHONPATH=python` works.
set_target_properties(_gpusonar PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/gpusonar")
