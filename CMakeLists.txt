cmake_minimum_required(VERSION 3.16)

project(adas_stack VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Optional TensorRT backend
#   cmake -DADAS_WITH_TENSORRT=ON \
#         -DTENSORRT_ROOT=/path/to/TensorRT \
#         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
# ---------------------------------------------------------------------------
option(ADAS_WITH_TENSORRT "Build the TensorRT inference backend" OFF)
option(ADAS_ENABLE_MCU_LINK "Enable MCU SPI communication link" OFF)

# Find Required Packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# For ASIO, look through system paths for the headers
set(_ASIO_BUNDLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/asio")

if(EXISTS "${_ASIO_BUNDLED_DIR}/asio.hpp")
    set(ASIO_INCLUDE_DIR "${_ASIO_BUNDLED_DIR}")
    message(STATUS "[ADAS] Asio: using bundled headers (${ASIO_INCLUDE_DIR})")
else()
    find_path(ASIO_INCLUDE_DIR
        NAMES asio.hpp
        PATHS
            /usr/include
            /usr/local/include
            /opt/local/include
        DOC "Path to standalone asio headers"
    )

    if(ASIO_INCLUDE_DIR)
        message(STATUS "[ADAS] Asio: found system headers (${ASIO_INCLUDE_DIR})")
    else()
        message(STATUS "[ADAS] Asio: not found locally -- fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(asio_fetch
            URL      https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.tar.gz
        )
        FetchContent_MakeAvailable(asio_fetch)
        set(ASIO_INCLUDE_DIR "${asio_fetch_SOURCE_DIR}/asio/include")
        message(STATUS "[ADAS] Asio: fetched to ${ASIO_INCLUDE_DIR}")
    endif()
endif()

# INTERFACE target so all dependents just link adas_asio
add_library(adas_asio INTERFACE)
target_include_directories(adas_asio INTERFACE "${ASIO_INCLUDE_DIR}")
target_compile_definitions(adas_asio INTERFACE ASIO_STANDALONE _WEBSOCKETPP_CPP11_STL_)
# Windows requires an explicit minimum platform version for asio socket APIs
if(WIN32)
    target_compile_definitions(adas_asio INTERFACE _WIN32_WINNT=0x0601)
endif()

# Project Library Target
add_library(adas_stack_lib
    src/config/config.cpp
    src/perception/lane_detection.cpp
    src/perception/road_object.cpp
    src/perception/perceptionengine.cpp
    src/calibration/extrinsic_calibration.cpp
    src/calibration/intrinsic_calibration.cpp
    src/yolo/opencv_dnn_engine.cpp
    src/streaming/websocket_streamer.cpp
    src/planning/planner.cpp
    src/controller/stanley_controller.cpp
    src/core/orchestrator.cpp
)

target_include_directories(adas_stack_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(adas_stack_lib PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    adas_asio
)

# ---------------------------------------------------------------------------
# TensorRT backend (only compiled when ADAS_WITH_TENSORRT=ON)
# ---------------------------------------------------------------------------
if(ADAS_WITH_TENSORRT)
    # --- CUDA ---------------------------------------------------------------
    find_package(CUDAToolkit REQUIRED)

    # --- TensorRT headers & libraries ---------------------------------------
    # Set TENSORRT_ROOT on the cmake command line if TRT is not in a standard
    # system path, e.g.:  -DTENSORRT_ROOT=/opt/TensorRT-10.x
    if(NOT DEFINED TENSORRT_ROOT)
        set(TENSORRT_ROOT "" CACHE PATH "Root directory of the TensorRT installation")
    endif()

    find_path(TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS ${TENSORRT_ROOT}/include
        PATH_SUFFIXES include
    )

    find_library(TENSORRT_NVINFER_LIB
        NAMES nvinfer
        HINTS ${TENSORRT_ROOT}/lib
        PATH_SUFFIXES lib lib64
    )

    find_library(TENSORRT_NVONNXPARSER_LIB
        NAMES nvonnxparser
        HINTS ${TENSORRT_ROOT}/lib
        PATH_SUFFIXES lib lib64
    )

    if(NOT TENSORRT_INCLUDE_DIR OR
       NOT TENSORRT_NVINFER_LIB   OR
       NOT TENSORRT_NVONNXPARSER_LIB)
        message(FATAL_ERROR
            "[ADAS] ADAS_WITH_TENSORRT=ON but TensorRT libraries were not found.\n"
            "  Set -DTENSORRT_ROOT=/path/to/TensorRT on the cmake command line.")
    endif()

    message(STATUS "[ADAS] TensorRT include : ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "[ADAS] nvinfer          : ${TENSORRT_NVINFER_LIB}")
    message(STATUS "[ADAS] nvonnxparser     : ${TENSORRT_NVONNXPARSER_LIB}")

    # Add the TRT source to the library
    target_sources(adas_stack_lib PRIVATE src/yolo/tensorrt_engine.cpp)

    target_include_directories(adas_stack_lib PUBLIC
        ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(adas_stack_lib PUBLIC
        CUDA::cudart
        ${TENSORRT_NVINFER_LIB}
        ${TENSORRT_NVONNXPARSER_LIB}
    )

    target_compile_definitions(adas_stack_lib PUBLIC ADAS_WITH_TENSORRT)

    message(STATUS "[ADAS] TensorRT backend enabled.")
else()
    message(STATUS "[ADAS] TensorRT backend disabled (pass -DADAS_WITH_TENSORRT=ON to enable).")
endif()

# ---------------------------------------------------------------------------
# Optional MCU Link SPI communication (only compiled when ADAS_ENABLE_MCU_LINK=ON)
# ---------------------------------------------------------------------------
if(ADAS_ENABLE_MCU_LINK)
    target_sources(adas_stack_lib PRIVATE src/mcu_link/mcu_link.cpp)
    target_compile_definitions(adas_stack_lib PUBLIC ADAS_ENABLE_MCU_LINK)
    message(STATUS "[ADAS] MCU Link enabled.")
else()
    message(STATUS "[ADAS] MCU Link disabled (pass -DADAS_ENABLE_MCU_LINK=ON to enable).")
endif()

# Compiler Warnings
if(MSVC)
    target_compile_options(adas_stack_lib PRIVATE /W4 /bigobj /FS /Zc:__cplusplus)
else()
    target_compile_options(adas_stack_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Project Executable Target
add_executable(adas_stack
    src/main.cpp
)

target_link_libraries(adas_stack PRIVATE adas_stack_lib)

# Test Executable Targets
add_executable(test_opencv_dnn_engine
    tests/test_opencv_dnn_engine.cpp
)
target_link_libraries(test_opencv_dnn_engine PRIVATE adas_stack_lib)

add_executable(test_stanley_controller
    tests/test_stanley_controller.cpp
)
target_link_libraries(test_stanley_controller PRIVATE adas_stack_lib)

