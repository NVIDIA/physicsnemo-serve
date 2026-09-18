cmake_minimum_required(VERSION 3.24)
project(PhysicsNeMoTransolverWorkflow LANGUAGES CXX CUDA)

option(PNMIR_ENABLE_AOTI "Run AOTInductor model packages" ON)
option(PNMIR_ENABLE_TENSORRT "Run TensorRT model packages" OFF)
set(PNMIR_WARP_ROOT "" CACHE PATH "Qualified Warp 1.15.0 package directory")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(backends)
if(PNMIR_ENABLE_AOTI)
  list(APPEND backends aoti)
endif()
if(PNMIR_ENABLE_TENSORRT)
  list(APPEND backends tensorrt)
endif()
if(NOT backends)
  message(FATAL_ERROR "Enable at least one model backend: AOTI or TensorRT")
endif()
find_package(PhysicsNeMoInference 0.1 CONFIG REQUIRED COMPONENTS ${backends})
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(VTK REQUIRED COMPONENTS
  CommonCore CommonDataModel FiltersCore IOGeometry IOXML)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_path(warp_include NAMES warp.h
  PATHS "${PNMIR_WARP_ROOT}/native" REQUIRED NO_DEFAULT_PATH)
if(NOT EXISTS "${PNMIR_WARP_ROOT}/bin/warp.so")
  message(FATAL_ERROR "PNMIR_WARP_ROOT must contain native/warp.h and bin/warp.so")
endif()
add_library(warp_runtime INTERFACE)
target_link_directories(warp_runtime INTERFACE "${PNMIR_WARP_ROOT}/bin")
target_link_libraries(warp_runtime INTERFACE "-l:warp.so")

add_library(transolver_workflow src/workflow.cpp src/warp_sdf.cu)
target_include_directories(transolver_workflow PUBLIC include PRIVATE "${warp_include}")
target_link_libraries(transolver_workflow
  PUBLIC PhysicsNeMoInference::runtime torch
  PRIVATE CUDA::cudart nlohmann_json::nlohmann_json warp_runtime ${VTK_LIBRARIES})
target_compile_definitions(transolver_workflow PUBLIC PNMIR_WARP_VERSION="1.15.0")
separate_arguments(torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(transolver_workflow PUBLIC ${torch_cxx_flags}
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)

add_executable(physicsnemo-transolver src/main.cpp)
target_link_libraries(physicsnemo-transolver
  PRIVATE transolver_workflow nlohmann_json::nlohmann_json VTK::CommonCore)
foreach(backend IN LISTS backends)
  string(TOUPPER "${backend}" backend_upper)
  target_compile_definitions(physicsnemo-transolver PRIVATE PNMIR_HAS_${backend_upper}=1)
  target_link_libraries(physicsnemo-transolver PRIVATE PhysicsNeMoInference::${backend})
endforeach()
set_property(TARGET physicsnemo-transolver APPEND PROPERTY
  BUILD_RPATH "${PNMIR_WARP_ROOT}/bin")

include(CTest)
if(BUILD_TESTING)
  add_executable(transolver_workflow_tests tests/workflow_test.cpp)
  target_link_libraries(transolver_workflow_tests PRIVATE transolver_workflow)
  add_test(NAME transolver_workflow_tests COMMAND transolver_workflow_tests)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME transolver_cli_help COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_help.py" $<TARGET_FILE:physicsnemo-transolver>)
endif()
