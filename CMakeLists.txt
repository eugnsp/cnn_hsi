cmake_minimum_required(VERSION 3.13)
project(cnn_hsi)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cnn_hsi "src/cnn_hsi.cpp")
target_compile_features(cnn_hsi PUBLIC cxx_std_17)
target_compile_options(cnn_hsi PUBLIC -Wall -Wpedantic -Wextra -Werror=return-type -m64 -march=native 
					   $<$<CONFIG:DEBUG>:-O0 -g> $<$<CONFIG:RELEASE>:-Wno-unused-parameter -Wno-deprecated-declarations -O3 -DNDEBUG>)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/eslib")
target_link_libraries(cnn_hsi eslib Threads::Threads)
