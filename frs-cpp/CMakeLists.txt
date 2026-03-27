cmake_minimum_required(VERSION 3.18)
project(frs_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ── TensorRT ──────────────────────────────────────────────────────────────────
set(TRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT root")
find_path(TRT_INCLUDE NvInfer.h HINTS ${TRT_ROOT}/include /usr/include)
find_library(TRT_LIB     nvinfer        HINTS ${TRT_ROOT}/lib /usr/lib/aarch64-linux-gnu)
find_library(TRT_PLUGIN  nvinfer_plugin HINTS ${TRT_ROOT}/lib /usr/lib/aarch64-linux-gnu)
find_library(TRT_ONNX    nvonnxparser   HINTS ${TRT_ROOT}/lib /usr/lib/aarch64-linux-gnu)

if(NOT TRT_INCLUDE OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TRT_ROOT.")
endif()
include_directories(${TRT_INCLUDE})

# ── OpenCV (Jetson build with GStreamer support) ───────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── libcurl ───────────────────────────────────────────────────────────────────
find_package(CURL REQUIRED)

# ── nlohmann/json (header-only) ───────────────────────────────────────────────
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)
endif()

# ── spdlog (header-only) ──────────────────────────────────────────────────────
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(spdlog
        URL https://github.com/gabime/spdlog/archive/v1.13.0.tar.gz)
    FetchContent_MakeAvailable(spdlog)
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(frs_runner
    src/main.cpp
    src/trt_engine.cpp
    src/face_detector.cpp
    src/face_embedder.cpp
    src/gst_capture.cpp
    src/http_client.cpp
    src/token_manager.cpp
    src/enroll_server.cpp
    src/runner.cpp
)

target_include_directories(frs_runner PRIVATE include)

target_link_libraries(frs_runner PRIVATE
    ${TRT_LIB}
    ${TRT_PLUGIN}
    ${TRT_ONNX}
    ${CUDA_LIBRARIES}
    cuda
    ${OpenCV_LIBS}
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    pthread
)

# Jetson aarch64 — enable NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(frs_runner PRIVATE -march=armv8-a+simd)
endif()

install(TARGETS frs_runner DESTINATION /usr/local/bin)
