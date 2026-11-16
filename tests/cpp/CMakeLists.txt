include(GoogleTest)

function(sonar_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE sonar::core sonar::io sonar::sim GTest::gtest GTest::gtest_main)
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

sonar_add_test(test_geometry test_geometry.cpp)
sonar_add_test(test_record_io test_record_io.cpp test_support.cpp)
target_include_directories(test_record_io PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_record_io PRIVATE ${HDF5_LIBRARIES})
sonar_add_test(test_allocator test_allocator.cpp)
