cmake_minimum_required(VERSION 3.24)
project(gpusonar VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(SONAR_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON or OFF")
set_property(CACHE SONAR_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(SONAR_BUILD_TESTS "Build the GoogleTest suites" ON)
option(SONAR_BUILD_PYTHON "Build the pybind11 module" ON)
option(SONAR_BUILD_HOLOSCAN "Build the Holoscan operators (needs the Holoscan SDK)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# CUDA backend. The kernels are written once and compiled for host and device;
# see docs/DECISIONS.md A-03.
# ---------------------------------------------------------------------------
set(SONAR_HAVE_CUDA OFF)
if(NOT SONAR_ENABLE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(SONAR_HAVE_CUDA ON)
  elseif(SONAR_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "SONAR_ENABLE_CUDA=ON but no CUDA compiler was found")
  endif()
endif()
message(STATUS "gpusonar: CUDA backend ${SONAR_HAVE_CUDA}")

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(sonar_warnings INTERFACE)
target_compile_options(sonar_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
add_subdirectory(libs/core)
add_subdirectory(libs/io)
add_subdirectory(libs/sim)

if(SONAR_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_dir}")
  add_subdirectory(python/bindings)
endif()

if(SONAR_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests/cpp)
  if(SONAR_BUILD_PYTHON)
    add_test(NAME pytest
      COMMAND "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests/python"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
  endif()
endif()
