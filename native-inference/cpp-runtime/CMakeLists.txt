cmake_minimum_required(VERSION 3.20)

project(PhysicsNeMoInference VERSION 0.1.0 LANGUAGES CXX)

option(PNMIR_BUILD_TESTS "Build PhysicsNeMo Inference tests" ON)
option(PNMIR_ENABLE_AOTI "Build the PyTorch AOTInductor backend" OFF)
option(PNMIR_BUILD_DOMINO_EXACT_OPS
  "Build Tensor-only exact ATen boundaries for DoMINO AOTI packages" OFF)
if(PNMIR_BUILD_DOMINO_EXACT_OPS AND NOT PNMIR_ENABLE_AOTI)
  message(FATAL_ERROR "PNMIR_BUILD_DOMINO_EXACT_OPS requires PNMIR_ENABLE_AOTI=ON")
endif()
option(PNMIR_ENABLE_ONNXRUNTIME "Build the ONNX Runtime backend" OFF)
option(
  PNMIR_ENABLE_ONNXRUNTIME_CUDA
  "Build ONNX Runtime CUDA device-I/O support tools and tests"
  OFF
)
option(PNMIR_ENABLE_TENSORRT "Build the NVIDIA TensorRT backend" OFF)
option(PNMIR_ENABLE_TENSORRT_EXACT
  "Build native exact TensorRT operators" OFF)
set(PNMIR_PYTORCH_SOURCE_ROOT "" CACHE PATH
  "Pinned PyTorch source tree supplying exact attention kernel headers")
set(PNMIR_CUTLASS_INCLUDE_DIR "" CACHE PATH
  "Pinned CUTLASS include tree matching the PyTorch attention kernel")
set(PNMIR_TORCH_INCLUDE_DIR "" CACHE PATH
  "Matching installed Torch include tree supplying generated headers")

if(PNMIR_ENABLE_TENSORRT_EXACT)
  if(NOT PNMIR_ENABLE_TENSORRT)
    message(FATAL_ERROR
      "PNMIR_ENABLE_TENSORRT_EXACT requires PNMIR_ENABLE_TENSORRT=ON")
  endif()
  if(NOT EXISTS "${PNMIR_PYTORCH_SOURCE_ROOT}/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h")
    message(FATAL_ERROR
      "PNMIR_PYTORCH_SOURCE_ROOT must contain the pinned PyTorch attention kernel headers")
  endif()
  if(NOT EXISTS "${PNMIR_CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    message(FATAL_ERROR
      "PNMIR_CUTLASS_INCLUDE_DIR must contain the matching cutlass headers")
  endif()
  if(NOT IS_DIRECTORY "${PNMIR_TORCH_INCLUDE_DIR}/torch")
    message(FATAL_ERROR
      "PNMIR_TORCH_INCLUDE_DIR must contain the matching installed Torch headers")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)
endif()

set(
  PNMIR_ONNXRUNTIME_ROOT
  ""
  CACHE PATH
  "Root of an ONNX Runtime C/C++ release containing include and lib"
)
set(
  PNMIR_TENSORRT_ROOT
  ""
  CACHE PATH
  "Root of a TensorRT C/C++ installation containing include and lib"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  # Export the C++ SDK symbols and avoid Windows SDK min/max macro collisions.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(PNMIR_ENABLE_ONNXRUNTIME_CUDA AND NOT PNMIR_ENABLE_ONNXRUNTIME)
  message(
    FATAL_ERROR
    "PNMIR_ENABLE_ONNXRUNTIME_CUDA requires PNMIR_ENABLE_ONNXRUNTIME=ON"
  )
endif()

include(GNUInstallDirs)
include(FetchContent)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  set(pnmir_download_timestamp_options)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(pnmir_download_timestamp_options DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  endif()
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    ${pnmir_download_timestamp_options}
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()
find_package(Threads REQUIRED)

add_library(pnmir
  src/backend.cpp
  src/manifest.cpp
  src/mock_backend.cpp
  src/prepared_state.cpp
  src/runtime.cpp
  src/tensor.cpp
  src/v1_api.cpp
)
add_library(PhysicsNeMoInference::runtime ALIAS pnmir)
set_target_properties(pnmir PROPERTIES OUTPUT_NAME physicsnemo_inference EXPORT_NAME runtime)
target_compile_features(pnmir PUBLIC cxx_std_20)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(pnmir PUBLIC PNMIR_SHARED=1)
endif()

target_include_directories(pnmir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  pnmir
  PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json> Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(pnmir PRIVATE -Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
  target_compile_options(pnmir PRIVATE /W4 /WX)
endif()

add_executable(pnmir_cli app/main.cpp)
set_target_properties(pnmir_cli PROPERTIES OUTPUT_NAME physicsnemo-infer EXPORT_NAME cli)
target_link_libraries(pnmir_cli PRIVATE nlohmann_json::nlohmann_json)

if(PNMIR_ENABLE_AOTI)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
    OUTPUT_VARIABLE PNMIR_TORCH_PYTHON_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import torch; print(torch.__version__.split('+', maxsplit=1)[0])"
    OUTPUT_VARIABLE PNMIR_TORCH_RUNTIME_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(
    PNMIR_TORCH_LIBRARY
    NAMES torch
    PATHS "${PNMIR_TORCH_PYTHON_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_library(
    PNMIR_C10_LIBRARY
    NAMES c10
    PATHS "${PNMIR_TORCH_PYTHON_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  set(
    TORCH_LIBRARY
    "${PNMIR_TORCH_LIBRARY}"
    CACHE FILEPATH "LibTorch selected by PhysicsNeMo Inference" FORCE
  )
  set(
    c10_LIBRARY
    "${PNMIR_C10_LIBRARY}"
    CACHE FILEPATH "c10 selected by PhysicsNeMo Inference" FORCE
  )
  set(
    Caffe2_DIR
    "${PNMIR_TORCH_PYTHON_ROOT}/share/cmake/Caffe2"
    CACHE PATH "Caffe2 selected by PhysicsNeMo Inference" FORCE
  )
  find_package(
    Torch REQUIRED
    PATHS "${PNMIR_TORCH_PYTHON_ROOT}/share/cmake/Torch"
    NO_DEFAULT_PATH
  )
  get_filename_component(
    PNMIR_PYTHON_BIN_DIR "${Python3_EXECUTABLE}" DIRECTORY
  )
  get_filename_component(
    PNMIR_PYTHON_PREFIX "${PNMIR_PYTHON_BIN_DIR}" DIRECTORY
  )

  add_library(pnmir_aoti src/aoti_backend.cpp)
  add_library(PhysicsNeMoInference::aoti ALIAS pnmir_aoti)
  set_target_properties(pnmir_aoti PROPERTIES OUTPUT_NAME physicsnemo_inference_aoti EXPORT_NAME aoti)
  target_include_directories(pnmir_aoti
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries(pnmir_aoti PUBLIC PhysicsNeMoInference::runtime PRIVATE torch)
  target_compile_definitions(
    pnmir_aoti
    PRIVATE PNMIR_TORCH_RUNTIME_VERSION="${PNMIR_TORCH_RUNTIME_VERSION}"
  )
  if(TARGET torch_cuda)
    target_compile_definitions(
      pnmir_aoti PRIVATE PNMIR_TORCH_HAS_CUDA=1
    )
  endif()

  if(PNMIR_BUILD_DOMINO_EXACT_OPS)
    add_library(pnmir_domino_exact_ops SHARED src/domino_exact_ops.cpp)
    add_library(PhysicsNeMoInference::domino_exact_ops ALIAS pnmir_domino_exact_ops)
    set_target_properties(pnmir_domino_exact_ops PROPERTIES
      OUTPUT_NAME pnmir_domino_exact_ops EXPORT_NAME domino_exact_ops)
    target_compile_features(pnmir_domino_exact_ops PUBLIC cxx_std_17)
    target_include_directories(pnmir_domino_exact_ops PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(pnmir_domino_exact_ops PRIVATE torch)
    target_compile_definitions(pnmir_domino_exact_ops PRIVATE NDEBUG)
    if(MSVC)
      target_compile_options(pnmir_domino_exact_ops PRIVATE /O2 ${TORCH_CXX_FLAGS})
    else()
      target_compile_options(pnmir_domino_exact_ops PRIVATE -O3 ${TORCH_CXX_FLAGS})
    endif()
    target_link_libraries(pnmir_aoti PRIVATE pnmir_domino_exact_ops)
    target_compile_definitions(pnmir_aoti PRIVATE PNMIR_HAS_DOMINO_EXACT_OPS=1)
  endif()

  target_compile_definitions(pnmir_cli PRIVATE PNMIR_HAS_AOTI=1)
  target_link_libraries(pnmir_cli PRIVATE PhysicsNeMoInference::aoti)
  set_property(
    TARGET pnmir_cli APPEND PROPERTY
    BUILD_RPATH "${PNMIR_PYTHON_PREFIX}/lib"
  )
endif()

if(PNMIR_ENABLE_ONNXRUNTIME)
  if(NOT PNMIR_ONNXRUNTIME_ROOT)
    message(
      FATAL_ERROR
      "PNMIR_ONNXRUNTIME_ROOT is required when ONNX Runtime is enabled"
    )
  endif()
  find_path(
    PNMIR_ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${PNMIR_ONNXRUNTIME_ROOT}/include"
    REQUIRED
    NO_DEFAULT_PATH
  )
  find_library(
    PNMIR_ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${PNMIR_ONNXRUNTIME_ROOT}/lib"
    REQUIRED
    NO_DEFAULT_PATH
  )

  add_library(pnmir_onnxruntime src/onnxruntime_backend.cpp)
  add_library(PhysicsNeMoInference::onnxruntime ALIAS pnmir_onnxruntime)
  set_target_properties(pnmir_onnxruntime PROPERTIES OUTPUT_NAME physicsnemo_inference_onnxruntime EXPORT_NAME onnxruntime)
  target_include_directories(pnmir_onnxruntime
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      "${PNMIR_ONNXRUNTIME_INCLUDE_DIR}"
  )
  target_link_libraries(
    pnmir_onnxruntime
    PUBLIC PhysicsNeMoInference::runtime
      $<BUILD_INTERFACE:${PNMIR_ONNXRUNTIME_LIBRARY}>
      $<INSTALL_INTERFACE:PhysicsNeMoInference::onnxruntime_native>
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      pnmir_onnxruntime PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  elseif(MSVC)
    target_compile_options(pnmir_onnxruntime PRIVATE /W4 /WX)
  endif()

  target_compile_definitions(pnmir_cli PRIVATE PNMIR_HAS_ONNXRUNTIME=1)
  target_link_libraries(pnmir_cli PRIVATE PhysicsNeMoInference::onnxruntime)
  get_filename_component(
    PNMIR_ONNXRUNTIME_LIBRARY_DIR
    "${PNMIR_ONNXRUNTIME_LIBRARY}"
    DIRECTORY
  )
  set_property(
    TARGET pnmir_cli APPEND PROPERTY
    BUILD_RPATH "${PNMIR_ONNXRUNTIME_LIBRARY_DIR}"
  )
endif()

if(PNMIR_ENABLE_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  find_path(
    PNMIR_TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
      "${PNMIR_TENSORRT_ROOT}/include"
      "${PNMIR_TENSORRT_ROOT}/include/x86_64-linux-gnu"
    REQUIRED
  )
  find_library(
    PNMIR_TENSORRT_LIBRARY
    NAMES nvinfer nvinfer_10
    HINTS
      "${PNMIR_TENSORRT_ROOT}/lib"
      "${PNMIR_TENSORRT_ROOT}/lib64"
      "${PNMIR_TENSORRT_ROOT}/lib/x86_64-linux-gnu"
    REQUIRED
  )
  find_library(
    PNMIR_TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin nvinfer_plugin_10
    HINTS
      "${PNMIR_TENSORRT_ROOT}/lib"
      "${PNMIR_TENSORRT_ROOT}/lib64"
      "${PNMIR_TENSORRT_ROOT}/lib/x86_64-linux-gnu"
    REQUIRED
  )

  add_library(pnmir_tensorrt src/tensorrt/tensorrt_backend.cpp)
  add_library(PhysicsNeMoInference::tensorrt ALIAS pnmir_tensorrt)
  set_target_properties(pnmir_tensorrt PROPERTIES OUTPUT_NAME physicsnemo_inference_tensorrt EXPORT_NAME tensorrt)
  target_include_directories(pnmir_tensorrt
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_include_directories(pnmir_tensorrt SYSTEM PRIVATE
    "${PNMIR_TENSORRT_INCLUDE_DIR}"
  )
  target_link_libraries(
    pnmir_tensorrt
    PUBLIC PhysicsNeMoInference::runtime
      $<BUILD_INTERFACE:${PNMIR_TENSORRT_LIBRARY}>
      $<BUILD_INTERFACE:${PNMIR_TENSORRT_PLUGIN_LIBRARY}>
      $<INSTALL_INTERFACE:PhysicsNeMoInference::tensorrt_native>
      $<INSTALL_INTERFACE:PhysicsNeMoInference::tensorrt_plugin_native>
    PRIVATE CUDA::cudart
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      pnmir_tensorrt PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  elseif(MSVC)
    target_compile_options(pnmir_tensorrt PRIVATE /W4 /WX)
  endif()

  target_compile_definitions(pnmir_cli PRIVATE PNMIR_HAS_TENSORRT=1)
  target_link_libraries(pnmir_cli PRIVATE PhysicsNeMoInference::tensorrt)
  get_filename_component(
    PNMIR_TENSORRT_LIBRARY_DIR
    "${PNMIR_TENSORRT_LIBRARY}"
    DIRECTORY
  )
  set_property(
    TARGET pnmir_cli APPEND PROPERTY
    BUILD_RPATH "${PNMIR_TENSORRT_LIBRARY_DIR}"
  )
  if(PNMIR_ENABLE_TENSORRT_EXACT)
    include(cmake/TensorRTExact.cmake)
  endif()
endif()

if(
  NOT PNMIR_ENABLE_AOTI AND
  NOT PNMIR_ENABLE_ONNXRUNTIME AND
  NOT PNMIR_ENABLE_TENSORRT
)
  target_link_libraries(pnmir_cli PRIVATE PhysicsNeMoInference::runtime)
endif()

if(PNMIR_ENABLE_TENSORRT OR PNMIR_ENABLE_ONNXRUNTIME_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(pnmir_device_runner app/device_runner.cpp)
  set_target_properties(
    pnmir_device_runner PROPERTIES OUTPUT_NAME physicsnemo-infer-device-runner
  )
  target_link_libraries(
    pnmir_device_runner PRIVATE CUDA::cudart
  )
  if(PNMIR_ENABLE_AOTI)
    target_compile_definitions(
      pnmir_device_runner PRIVATE PNMIR_HAS_AOTI=1
    )
    target_link_libraries(pnmir_device_runner PRIVATE PhysicsNeMoInference::aoti)
  endif()
  if(PNMIR_ENABLE_ONNXRUNTIME)
    target_compile_definitions(
      pnmir_device_runner PRIVATE PNMIR_HAS_ONNXRUNTIME=1
    )
    target_link_libraries(
      pnmir_device_runner PRIVATE PhysicsNeMoInference::onnxruntime
    )
  endif()
  if(PNMIR_ENABLE_TENSORRT)
    target_compile_definitions(
      pnmir_device_runner PRIVATE PNMIR_HAS_TENSORRT=1
    )
    target_link_libraries(
      pnmir_device_runner PRIVATE PhysicsNeMoInference::tensorrt
    )
  endif()
endif()

if(PNMIR_BUILD_TESTS)
  include(cmake/Tests.cmake)
endif()

if(APPLE)
  set(pnmir_install_rpath "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  set(pnmir_install_rpath "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
foreach(pnmir_target IN ITEMS pnmir pnmir_cli pnmir_aoti pnmir_domino_exact_ops pnmir_onnxruntime pnmir_tensorrt pnmir_device_runner ${pnmir_tensorrt_exact_targets})
  if(TARGET ${pnmir_target})
    set_target_properties(${pnmir_target} PROPERTIES
      INSTALL_RPATH "${pnmir_install_rpath}"
      INSTALL_RPATH_USE_LINK_PATH FALSE
    )
  endif()
endforeach()

include(CMakePackageConfigHelpers)
set(pnmir_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/PhysicsNeMoInference")
configure_package_config_file(cmake/PhysicsNeMoInferenceConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/PhysicsNeMoInferenceConfig.cmake"
  INSTALL_DESTINATION "${pnmir_cmake_dir}"
)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/PhysicsNeMoInferenceConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)

install(TARGETS pnmir pnmir_cli EXPORT PhysicsNeMoInferenceCoreTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if(PNMIR_ENABLE_AOTI)
  install(TARGETS pnmir_aoti EXPORT PhysicsNeMoInferenceAotiTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  if(TARGET pnmir_domino_exact_ops)
    install(TARGETS pnmir_domino_exact_ops EXPORT PhysicsNeMoInferenceAotiTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  install(EXPORT PhysicsNeMoInferenceAotiTargets FILE PhysicsNeMoInferenceAotiTargets.cmake NAMESPACE PhysicsNeMoInference::
    DESTINATION "${pnmir_cmake_dir}")
endif()
if(PNMIR_ENABLE_ONNXRUNTIME)
  install(TARGETS pnmir_onnxruntime EXPORT PhysicsNeMoInferenceOnnxruntimeTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(EXPORT PhysicsNeMoInferenceOnnxruntimeTargets FILE PhysicsNeMoInferenceOnnxruntimeTargets.cmake NAMESPACE PhysicsNeMoInference::
    DESTINATION "${pnmir_cmake_dir}")
endif()
if(PNMIR_ENABLE_TENSORRT)
  install(TARGETS pnmir_tensorrt ${pnmir_tensorrt_exact_targets} EXPORT PhysicsNeMoInferenceTensorrtTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(EXPORT PhysicsNeMoInferenceTensorrtTargets FILE PhysicsNeMoInferenceTensorrtTargets.cmake NAMESPACE PhysicsNeMoInference::
    DESTINATION "${pnmir_cmake_dir}")
endif()
if(TARGET pnmir_device_runner)
  install(
    TARGETS pnmir_device_runner
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT PhysicsNeMoInferenceCoreTargets FILE PhysicsNeMoInferenceCoreTargets.cmake NAMESPACE PhysicsNeMoInference::
  DESTINATION "${pnmir_cmake_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PhysicsNeMoInferenceConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/PhysicsNeMoInferenceConfigVersion.cmake"
  DESTINATION "${pnmir_cmake_dir}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/physicsnemo-inference")
