cmake_minimum_required(VERSION 3.13)
project(cnn_hsi)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MKL_LIB "$ENV{MKLROOT}/lib/intel64")

add_executable(cnn_hsi "src/cnn_hsi.cpp")
target_compile_features(cnn_hsi PUBLIC cxx_std_17)
target_compile_options(cnn_hsi PUBLIC -Wall -Wpedantic -Wextra -Werror=return-type -m64 -march=native 
					   $<$<CONFIG:DEBUG>:-O0 -g> $<$<CONFIG:RELEASE>:-Wno-unused-parameter -Wno-deprecated-declarations -O3 -DNDEBUG>)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	target_compile_options(cnn_hsi PUBLIC -Wno-psabi)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang")
	target_compile_options(cnn_hsi PUBLIC -ferror-limit=5)
else()
	target_compile_options(cnn_hsi PUBLIC -fmax-errors=5)
endif()

target_link_directories(cnn_hsi PRIVATE "${MKL_LIB}")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/esu")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/esl")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

target_link_libraries(cnn_hsi esl esu Threads::Threads)
