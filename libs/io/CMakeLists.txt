add_library(sonar_io
  src/record_file.cpp
  src/validate.cpp)
add_library(sonar::io ALIAS sonar_io)
target_include_directories(sonar_io PUBLIC include PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(sonar_io PUBLIC sonar::core PRIVATE ${HDF5_LIBRARIES} sonar_warnings)
