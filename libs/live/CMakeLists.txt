add_library(sonar_live src/cluster.cpp src/tracker.cpp src/classify.cpp src/doa.cpp src/chain.cpp src/udp_source.cpp src/sync.cpp)
add_library(sonar::live ALIAS sonar_live)
target_include_directories(sonar_live PUBLIC include)
target_link_libraries(sonar_live PUBLIC sonar::core sonar::dsp sonar::recon sonar::detect Eigen3::Eigen PRIVATE sonar_warnings)
