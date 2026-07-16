cmake_minimum_required(VERSION 3.18)

project(fast_foundation_stereo_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20015>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20013>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20091>
)

find_package(CUDAToolkit REQUIRED)

# ---- TensorRT ----
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation path")
set(TENSORRT_HINT_DIRS
  ${TENSORRT_ROOT}
  $ENV{CONDA_PREFIX}
  /usr
  /usr/local
  /usr/local/tensorrt
  ${CUDAToolkit_TARGET_DIR}
  /usr/local/cuda
  /usr/local/cuda/targets/x86_64-linux
  /usr/local/cuda-13.0
  /usr/local/cuda-13.0/targets/x86_64-linux
  /usr/src/tensorrt
)
if(DEFINED ENV{CONDA_PREFIX})
  file(GLOB TENSORRT_PY_LIB_DIRS
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/tensorrt_libs"
  )
  file(GLOB TENSORRT_PY_INCLUDE_DIRS
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/tensorrt/include"
  )
endif()

find_library(NVINFER_LIB
  NAMES nvinfer nvinfer.so.10
  HINTS
    ${TENSORRT_HINT_DIRS}
    ${TENSORRT_PY_LIB_DIRS}
  PATH_SUFFIXES
    lib
    lib64
    lib/x86_64-linux-gnu
)
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT nvinfer library not found. Set -DTENSORRT_ROOT=/path/to/tensorrt")
endif()
message(STATUS "TensorRT nvinfer: ${NVINFER_LIB}")

find_library(NVONNXPARSER_LIB
  NAMES nvonnxparser nvonnxparser.so.10
  HINTS
    ${TENSORRT_HINT_DIRS}
    ${TENSORRT_PY_LIB_DIRS}
  PATH_SUFFIXES
    lib
    lib64
    lib/x86_64-linux-gnu
)
if(NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR "TensorRT nvonnxparser library not found. Set -DTENSORRT_ROOT=/path/to/tensorrt")
endif()
message(STATUS "TensorRT nvonnxparser: ${NVONNXPARSER_LIB}")

find_path(NVINFER_INCLUDE_DIR NvInfer.h
  HINTS
    ${TENSORRT_HINT_DIRS}
    ${TENSORRT_PY_INCLUDE_DIRS}
  PATH_SUFFIXES
    include
    include/x86_64-linux-gnu
)
if(NOT NVINFER_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers (NvInfer.h) not found. Set -DTENSORRT_ROOT=/path/to/tensorrt")
endif()

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)

add_library(ffs_gwc_plugin SHARED
  src/gwc_volume_plugin.cpp
  src/depth_kernels.cu
)
set_target_properties(ffs_gwc_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  OUTPUT_NAME "ffs_gwc_plugin"
)
target_include_directories(ffs_gwc_plugin PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NVINFER_INCLUDE_DIR}
)
target_link_libraries(ffs_gwc_plugin PUBLIC
  ${NVINFER_LIB}
  CUDA::cudart
)

add_library(ffs_depth_inference STATIC
  src/ffs_depth_tensorrt.cpp
  src/ffs_depth_single_tensorrt.cpp
  src/gwc_volume_plugin.cpp
  src/depth_kernels.cu
)
set_target_properties(ffs_depth_inference PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(ffs_depth_inference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NVINFER_INCLUDE_DIR}
)
target_link_libraries(ffs_depth_inference PUBLIC
  ${NVINFER_LIB}
  CUDA::cudart
)

if(OpenCV_FOUND)
  add_executable(ffs_depth_main app/main.cpp)
  target_link_libraries(ffs_depth_main PRIVATE
    ffs_depth_inference
    ${OpenCV_LIBS}
  )
else()
  message(WARNING "OpenCV C++ development package not found; skipping ffs_depth_main and ffs_profile_speed")
endif()

add_executable(ffs_build_single_engine app/build_single_engine.cpp)
target_link_libraries(ffs_build_single_engine PRIVATE
  ffs_depth_inference
  ${NVONNXPARSER_LIB}
)

if(OpenCV_FOUND)
  add_executable(ffs_profile_speed app/profile_speed.cpp)
  target_link_libraries(ffs_profile_speed PRIVATE
    ffs_depth_inference
    ${OpenCV_LIBS}
  )
endif()
