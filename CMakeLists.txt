cmake_minimum_required(VERSION 3.16)

project(adas_stack VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Required Packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Project Library Target
add_library(adas_stack_lib
    src/config/config.cpp
    src/perception/lane_detection.cpp
    src/perception/road_object.cpp
    src/calibration/extrinsic_calibration.cpp
    src/calibration/intrinsic_calibration.cpp
    src/yolo/opencv_dnn_engine.cpp
)

target_include_directories(adas_stack_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(adas_stack_lib PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Compiler Warnings
if(MSVC)
    target_compile_options(adas_stack_lib PRIVATE /W4 /bigobj /FS)
else()
    target_compile_options(adas_stack_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Project Executable Target
add_executable(adas_stack
    src/main.cpp
)

target_link_libraries(adas_stack PRIVATE adas_stack_lib)

# Test Executable Target
add_executable(test_opencv_dnn_engine
    tests/test_opencv_dnn_engine.cpp
)

target_link_libraries(test_opencv_dnn_engine PRIVATE adas_stack_lib)

