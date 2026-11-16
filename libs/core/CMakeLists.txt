add_library(sonar_core
  src/allocator.cpp
  src/platform.cpp
  src/record.cpp)
add_library(sonar::core ALIAS sonar_core)
target_include_directories(sonar_core PUBLIC include)
target_compile_definitions(sonar_core PUBLIC SONAR_HAVE_CUDA=$<BOOL:${SONAR_HAVE_CUDA}>)
target_link_libraries(sonar_core PUBLIC Threads::Threads PRIVATE sonar_warnings)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sonar_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SONAR_HAVE_CUDA)
  target_link_libraries(sonar_core PUBLIC CUDA::cudart)
endif()
