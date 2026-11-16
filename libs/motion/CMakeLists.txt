add_library(sonar_motion src/dpc_solver.cpp)
add_library(sonar::motion ALIAS sonar_motion)
target_include_directories(sonar_motion PUBLIC include)
target_link_libraries(sonar_motion PUBLIC sonar::core PRIVATE Ceres::ceres sonar_warnings)
