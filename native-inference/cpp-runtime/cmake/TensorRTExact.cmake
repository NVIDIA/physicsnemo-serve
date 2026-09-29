# Keep these filenames stable: Model Builder loads the same native plugins.
set(pnmir_tensorrt_exact_targets)
foreach(operator IN ITEMS linear gemm token_sum slice_bmm deslice_bmm layer_norm softmax attention gelu weighted_blend scalar_div inverse_distance_blend)
  set(plugin pnmir_tensorrt_exact_${operator}_plugin)
  if(operator MATCHES "^(linear|gemm|slice_bmm|deslice_bmm)$")
    set(extension cpp)
  else()
    set(extension cu)
  endif()
  add_library(${plugin} SHARED src/tensorrt/tensorrt_exact_${operator}_plugin.${extension})
  set_target_properties(${plugin} PROPERTIES
    EXPORT_NAME tensorrt_exact_${operator}_plugin
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(${plugin}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_include_directories(${plugin} SYSTEM PRIVATE "${PNMIR_TENSORRT_INCLUDE_DIR}")
  if(MSVC AND extension STREQUAL "cu")
    # Visual Studio's CUDA task emits SYSTEM directories as ordinary -I flags.
    # Mark only the vendor headers external for nvcc's MSVC host compilation.
    target_compile_options(${plugin} PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:I\"${PNMIR_TENSORRT_INCLUDE_DIR}\">"
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:W0>)
  endif()
  target_link_libraries(${plugin} PRIVATE
    $<BUILD_INTERFACE:${PNMIR_TENSORRT_LIBRARY}>
    $<INSTALL_INTERFACE:PhysicsNeMoInference::tensorrt_native>
    CUDA::cudart)
  if(operator STREQUAL "linear")
    target_link_libraries(${plugin} PRIVATE CUDA::cublasLt)
  elseif(operator MATCHES "^(gemm|slice_bmm|deslice_bmm)$")
    target_link_libraries(${plugin} PRIVATE CUDA::cublas)
  endif()
  # Match the proven compiler flags; do not enable fast math or alter reductions.
  if(operator STREQUAL "attention")
    target_include_directories(${plugin} PRIVATE
      "${PNMIR_PYTORCH_SOURCE_ROOT}"
      "${PNMIR_PYTORCH_SOURCE_ROOT}/aten/src"
      "${PNMIR_CUTLASS_INCLUDE_DIR}"
      "${PNMIR_TORCH_INCLUDE_DIR}")
    target_compile_options(${plugin} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  elseif(extension STREQUAL "cu")
    if(MSVC)
      if(CMAKE_GENERATOR MATCHES "^Visual Studio")
        if(CMAKE_VERSION VERSION_LESS 3.24)
          message(FATAL_ERROR "Exact TensorRT plugins require CMake 3.24 or newer with Visual Studio")
        endif()
        # CUDA's MSBuild task otherwise appends /W1 after target options.
        set_property(SOURCE src/tensorrt/tensorrt_exact_${operator}_plugin.cu
          APPEND PROPERTY VS_SETTINGS "Warning=W4")
        target_compile_options(${plugin} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
      else()
        target_compile_options(${plugin} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/WX>)
      endif()
    else()
      target_compile_options(${plugin} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${plugin} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  elseif(MSVC)
    target_compile_options(${plugin} PRIVATE /W4 /WX)
  endif()
  target_link_libraries(pnmir_tensorrt PRIVATE ${plugin})
  list(APPEND pnmir_tensorrt_exact_targets ${plugin})
endforeach()
target_compile_definitions(pnmir_tensorrt PRIVATE PNMIR_HAS_TENSORRT_EXACT=1)
