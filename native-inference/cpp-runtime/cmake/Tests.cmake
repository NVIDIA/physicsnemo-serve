# Test registration is included in the SDK directory to preserve paths and properties.
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
  NAME pnmir_tensorrt_exact_configuration
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/configuration.py
    --sdk-source ${CMAKE_CURRENT_SOURCE_DIR} ExactConfigurationTests -v
)

add_test(
  NAME pnmir_windows_configuration
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/configuration.py
    --sdk-source ${CMAKE_CURRENT_SOURCE_DIR} WindowsConfigurationTests -v
)

add_test(
  NAME pnmir_install_consumer_portability
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_install_consumer_portability.py -v
)

add_test(
  NAME pnmir_device_runner_cuda_skip
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_device_runner_cuda_skip.py -v
)

add_executable(pnmir_device_runner_count_test tests/device_runner_count_test.cpp)
add_test(NAME pnmir_device_runner_count COMMAND pnmir_device_runner_count_test)

add_test(
  NAME pnmir_domino_plugin_paths
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_domino_plugin_paths.py -v
)

add_test(
  NAME pnmir_cli_output_metadata
  COMMAND ${Python3_EXECUTABLE} -S
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_output_metadata.py
    --pnmir $<TARGET_FILE:pnmir_cli> -v
)
foreach(linkage IN ITEMS static shared)
  set(install_test_options)
  if(linkage STREQUAL "shared")
    list(APPEND install_test_options --shared)
  endif()
  add_test(
    NAME pnmir_install_${linkage}_consumer
    COMMAND ${Python3_EXECUTABLE} -S
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer.py
      --sdk-source ${CMAKE_CURRENT_SOURCE_DIR} ${install_test_options} -v
  )
  foreach(component IN ITEMS aoti onnxruntime tensorrt)
    string(TOUPPER "${component}" component_option)
    if(PNMIR_ENABLE_${component_option})
      set(component_test_options --component ${component})
      if(component STREQUAL "aoti")
        list(APPEND component_test_options --torch-root "${PNMIR_TORCH_PYTHON_ROOT}")
      elseif(component STREQUAL "onnxruntime")
        list(APPEND component_test_options --onnxruntime-root "${PNMIR_ONNXRUNTIME_ROOT}")
      elseif(PNMIR_TENSORRT_ROOT)
        list(APPEND component_test_options --tensorrt-root "${PNMIR_TENSORRT_ROOT}")
      endif()
      if(CUDAToolkit_ROOT)
        list(APPEND component_test_options --cuda-root "${CUDAToolkit_ROOT}")
      elseif(CUDA_TOOLKIT_ROOT_DIR)
        list(APPEND component_test_options --cuda-root "${CUDA_TOOLKIT_ROOT_DIR}")
      endif()
      if(CMAKE_CUDA_ARCHITECTURES)
        string(REPLACE ";" "," cuda_test_architectures "${CMAKE_CUDA_ARCHITECTURES}")
        list(APPEND component_test_options --cuda-architectures "${cuda_test_architectures}")
      endif()
      add_test(
        NAME pnmir_install_${component}_${linkage}_consumer
        COMMAND ${Python3_EXECUTABLE} -S
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer.py
          --sdk-source ${CMAKE_CURRENT_SOURCE_DIR}
          ${install_test_options} ${component_test_options} -v
      )
    endif()
  endforeach()
endforeach()

add_executable(pnmir_tests tests/runtime_test.cpp)
target_link_libraries(pnmir_tests PRIVATE PhysicsNeMoInference::runtime)
target_compile_definitions(
  pnmir_tests PRIVATE PNMIR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME pnmir_tests COMMAND pnmir_tests)

add_executable(pnmir_v1_api_tests tests/v1_api_test.cpp)
target_link_libraries(pnmir_v1_api_tests PRIVATE PhysicsNeMoInference::runtime)
target_compile_definitions(
  pnmir_v1_api_tests PRIVATE PNMIR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME pnmir_v1_api_tests COMMAND pnmir_v1_api_tests)

add_test(
  NAME pnmir_cli_dynamic_values
  COMMAND
    pnmir_cli
    run
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/identity
    --values
    1,2,3.5
)
set_tests_properties(
  pnmir_cli_dynamic_values
  PROPERTIES PASS_REGULAR_EXPRESSION "output: 1 2 3.5"
)

if(TARGET pnmir_device_runner)
  add_executable(
    pnmir_persistent_device_state_test
    tests/persistent_device_state_test.cpp
  )
  target_link_libraries(
    pnmir_persistent_device_state_test
    PRIVATE CUDA::cudart
  )
  if(PNMIR_ENABLE_AOTI)
    target_compile_definitions(
      pnmir_persistent_device_state_test PRIVATE PNMIR_HAS_AOTI=1
    )
    target_link_libraries(
      pnmir_persistent_device_state_test PRIVATE PhysicsNeMoInference::aoti
    )
  endif()
  if(PNMIR_ENABLE_ONNXRUNTIME)
    target_compile_definitions(
      pnmir_persistent_device_state_test
      PRIVATE PNMIR_HAS_ONNXRUNTIME=1
    )
    target_link_libraries(
      pnmir_persistent_device_state_test PRIVATE PhysicsNeMoInference::onnxruntime
    )
  endif()
  if(PNMIR_ENABLE_TENSORRT)
    target_compile_definitions(
      pnmir_persistent_device_state_test
      PRIVATE PNMIR_HAS_TENSORRT=1
    )
    target_link_libraries(
      pnmir_persistent_device_state_test PRIVATE PhysicsNeMoInference::tensorrt
    )
  endif()
endif()

if(PNMIR_ENABLE_AOTI)
  if(TARGET torch_cuda)
    add_executable(
      pnmir_aoti_device_index_test tests/aoti_device_index_test.cpp
    )
    target_link_libraries(
      pnmir_aoti_device_index_test PRIVATE PhysicsNeMoInference::aoti torch
    )
    add_test(
      NAME pnmir_aoti_device_index
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/aoti_device_index.py
        --runner $<TARGET_FILE:pnmir_aoti_device_index_test>
        --output ${CMAKE_CURRENT_BINARY_DIR}/aoti-device-index.pnmir
    )
    set_tests_properties(
      pnmir_aoti_device_index PROPERTIES SKIP_RETURN_CODE 77
    )
  endif()

  add_executable(
    pnmir_aoti_owned_outputs_test tests/aoti_owned_outputs_test.cpp
  )
  target_link_libraries(
    pnmir_aoti_owned_outputs_test PRIVATE PhysicsNeMoInference::aoti torch
  )
  foreach(target_device IN ITEMS cpu cuda)
    add_test(
      NAME pnmir_aoti_owned_outputs_${target_device}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/aoti_owned_outputs.py
        --runner $<TARGET_FILE:pnmir_aoti_owned_outputs_test>
        --output ${CMAKE_CURRENT_BINARY_DIR}/aoti-owned-outputs-${target_device}.pnmir
        --target ${target_device}
    )
    add_test(
      NAME pnmir_aoti_dynamic_outputs_${target_device}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/aoti_dynamic_outputs.py
        --pnmir $<TARGET_FILE:pnmir_cli>
        --output ${CMAKE_CURRENT_BINARY_DIR}/aoti-dynamic-outputs-${target_device}.pnmir
        --target ${target_device}
    )
  endforeach()
  set_tests_properties(
    pnmir_aoti_owned_outputs_cuda pnmir_aoti_dynamic_outputs_cuda
    PROPERTIES SKIP_RETURN_CODE 77
  )

  add_test(
    NAME pnmir_aoti_integration
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/aoti_integration.py
      --pnmir $<TARGET_FILE:pnmir_cli>
      --output ${CMAKE_CURRENT_BINARY_DIR}/aoti-identity.pnmir
  )
  add_test(
    NAME pnmir_aoti_cuda_integration
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/aoti_integration.py
      --pnmir $<TARGET_FILE:pnmir_cli>
      --output ${CMAKE_CURRENT_BINARY_DIR}/aoti-identity-cuda.pnmir
      --target cuda
  )
  set_tests_properties(
    pnmir_aoti_cuda_integration PROPERTIES SKIP_RETURN_CODE 77
  )

  add_executable(
    pnmir_aoti_device_input_test tests/aoti_device_input_test.cpp
  )
  target_link_libraries(
    pnmir_aoti_device_input_test PRIVATE PhysicsNeMoInference::aoti torch
  )
  add_test(
    NAME pnmir_aoti_device_input
    COMMAND
      pnmir_aoti_device_input_test
      ${CMAKE_CURRENT_BINARY_DIR}/aoti-identity-cuda.pnmir
  )
  set_tests_properties(
    pnmir_aoti_device_input
    PROPERTIES
      DEPENDS pnmir_aoti_cuda_integration
      SKIP_RETURN_CODE 77
  )

  if(TARGET pnmir_persistent_device_state_test)
    add_test(
      NAME pnmir_aoti_persistent_device_state
      COMMAND
        pnmir_persistent_device_state_test
        aoti
        ${CMAKE_CURRENT_BINARY_DIR}/aoti-identity-cuda.pnmir
    )
    set_tests_properties(
      pnmir_aoti_persistent_device_state
      PROPERTIES
        DEPENDS pnmir_aoti_cuda_integration
        SKIP_RETURN_CODE 77
    )
  endif()
endif()

if(PNMIR_ENABLE_ONNXRUNTIME)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import onnx"
    RESULT_VARIABLE PNMIR_ONNX_IMPORT_RESULT
  )
  if(NOT PNMIR_ONNX_IMPORT_RESULT EQUAL 0)
    message(
      FATAL_ERROR
      "ONNX Runtime integration tests require the Python 'onnx' package"
    )
  endif()

  add_test(
    NAME pnmir_onnxruntime_integration
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/onnxruntime_integration.py
      --pnmir $<TARGET_FILE:pnmir_cli>
      --output ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-affine.pnmir
  )
  add_test(
    NAME pnmir_physicsnemo_channel_package
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/physicsnemo_channel_package.py
      --pnmir $<TARGET_FILE:pnmir_cli>
      --package
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/physicsnemo_cfd_channel/physicsnemo-channel-flow-v0.1.0-onnxruntime-cpu-fp32.pnmir
  )

  if(PNMIR_ENABLE_ONNXRUNTIME_CUDA)
    add_test(
      NAME pnmir_onnxruntime_device_runner
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/device_runner_integration.py
        --runner $<TARGET_FILE:pnmir_device_runner>
        --backend onnxruntime -v
    )
    add_test(
      NAME pnmir_onnxruntime_cuda_integration
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/onnxruntime_cuda_integration.py
        --pnmir $<TARGET_FILE:pnmir_cli>
        --output
          ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-cuda-affine.pnmir
    )
    add_executable(
      pnmir_onnxruntime_device_io_test
      tests/onnxruntime_device_io_test.cpp
    )
    target_link_libraries(
      pnmir_onnxruntime_device_io_test
      PRIVATE PhysicsNeMoInference::onnxruntime CUDA::cudart
    )
    add_test(
      NAME pnmir_onnxruntime_device_io
      COMMAND
        pnmir_onnxruntime_device_io_test
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-cuda-affine.pnmir
    )
    set_tests_properties(
      pnmir_onnxruntime_cuda_integration
      pnmir_onnxruntime_device_io
      PROPERTIES SKIP_RETURN_CODE 77
    )
    set_tests_properties(
      pnmir_onnxruntime_device_io
      PROPERTIES DEPENDS pnmir_onnxruntime_cuda_integration
    )

    if(PNMIR_ENABLE_AOTI)
      add_test(
        NAME pnmir_onnxruntime_exporter_integration
        COMMAND
          ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/onnxruntime_exporter_integration.py
          --pnmir $<TARGET_FILE:pnmir_cli>
          --output
            ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-exporter-affine.pnmir
      )
      set_tests_properties(
        pnmir_onnxruntime_exporter_integration
        PROPERTIES SKIP_RETURN_CODE 77
      )
    endif()

    if(TARGET pnmir_persistent_device_state_test)
      add_test(
        NAME pnmir_onnxruntime_persistent_device_state
        COMMAND
          pnmir_persistent_device_state_test
          onnxruntime
          ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-cuda-affine.pnmir
      )
      set_tests_properties(
        pnmir_onnxruntime_persistent_device_state
        PROPERTIES
          DEPENDS pnmir_onnxruntime_cuda_integration
          SKIP_RETURN_CODE 77
      )
    endif()
  endif()

  if(PNMIR_ENABLE_AOTI)
    add_test(
      NAME pnmir_multi_backend_integration
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/multi_backend_integration.py
        --pnmir $<TARGET_FILE:pnmir_cli>
        --output ${CMAKE_CURRENT_BINARY_DIR}/multi-backend-affine.pnmir
    )
  endif()
endif()

if(PNMIR_ENABLE_TENSORRT)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import onnx, tensorrt, torch"
    RESULT_VARIABLE PNMIR_TENSORRT_PYTHON_IMPORT_RESULT
  )
  if(NOT PNMIR_TENSORRT_PYTHON_IMPORT_RESULT EQUAL 0)
    message(
      FATAL_ERROR
      "TensorRT integration tests require Python onnx, tensorrt, and torch"
    )
  endif()

  add_test(
    NAME pnmir_tensorrt_integration
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorrt_integration.py
      --pnmir $<TARGET_FILE:pnmir_cli>
      --output ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-affine.pnmir
  )
  add_test(
    NAME pnmir_tensorrt_device_runner
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/device_runner_integration.py
      --runner $<TARGET_FILE:pnmir_device_runner>
      --backend tensorrt -v
  )
  set_tests_properties(
    pnmir_tensorrt_device_runner PROPERTIES SKIP_RETURN_CODE 77
  )
  if(PNMIR_ENABLE_TENSORRT_EXACT)
    add_executable(pnmir_tensorrt_exact_dimensions_test
      tests/tensorrt_exact_dimensions_test.cpp)
    target_include_directories(pnmir_tensorrt_exact_dimensions_test SYSTEM PRIVATE
      "${PNMIR_TENSORRT_INCLUDE_DIR}")
    target_link_libraries(pnmir_tensorrt_exact_dimensions_test PRIVATE
      pnmir_tensorrt_exact_gemm_plugin pnmir_tensorrt_exact_slice_bmm_plugin
      pnmir_tensorrt_exact_deslice_bmm_plugin
      pnmir_tensorrt_exact_layer_norm_plugin pnmir_tensorrt_exact_softmax_plugin
      "${PNMIR_TENSORRT_LIBRARY}" CUDA::cudart)
    add_test(NAME pnmir_tensorrt_exact_dimensions
      COMMAND pnmir_tensorrt_exact_dimensions_test)
    add_test(
      NAME pnmir_tensorrt_exact_linear
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorrt_exact_linear.py
        --plugin $<TARGET_FILE:pnmir_tensorrt_exact_linear_plugin>
        --output ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-exact-linear
    )
    set_tests_properties(pnmir_tensorrt_exact_linear
      PROPERTIES SKIP_RETURN_CODE 77)
    add_test(
      NAME pnmir_tensorrt_exact_deslice_bmm
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorrt_exact_deslice_bmm.py
        --pnmir $<TARGET_FILE:pnmir_cli>
        --plugin $<TARGET_FILE:pnmir_tensorrt_exact_deslice_bmm_plugin>
        --output ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-exact-deslice
    )
    set_tests_properties(pnmir_tensorrt_exact_deslice_bmm
      PROPERTIES SKIP_RETURN_CODE 77)
    add_test(
      NAME pnmir_tensorrt_exact_weighted_blend
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorrt_exact_weighted_blend.py
        --pnmir $<TARGET_FILE:pnmir_cli>
        --plugin $<TARGET_FILE:pnmir_tensorrt_exact_weighted_blend_plugin>
        --output ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-exact-weighted-blend
    )
    set_tests_properties(pnmir_tensorrt_exact_weighted_blend
      PROPERTIES SKIP_RETURN_CODE 77)
    if(PNMIR_BUILD_DOMINO_EXACT_OPS)
      add_test(
        NAME pnmir_domino_exact_integration
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/domino_exact_integration.py
          --pnmir $<TARGET_FILE:pnmir_cli>
          --sidecar $<TARGET_FILE:pnmir_domino_exact_ops>
          --scalar-div-plugin $<TARGET_FILE:pnmir_tensorrt_exact_scalar_div_plugin>
          --inverse-distance-blend-plugin $<TARGET_FILE:pnmir_tensorrt_exact_inverse_distance_blend_plugin>
          --output ${CMAKE_CURRENT_BINARY_DIR}/domino-exact-integration
      )
      set_tests_properties(pnmir_domino_exact_integration
        PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endif()
  add_executable(
    pnmir_tensorrt_device_io_test tests/tensorrt_device_io_test.cpp
  )
  target_link_libraries(
    pnmir_tensorrt_device_io_test
    PRIVATE PhysicsNeMoInference::tensorrt CUDA::cudart
  )
  add_test(
    NAME pnmir_tensorrt_device_io
    COMMAND
      pnmir_tensorrt_device_io_test
      ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-affine.pnmir
  )
  set_tests_properties(
    pnmir_tensorrt_integration
    pnmir_tensorrt_device_io
    PROPERTIES SKIP_RETURN_CODE 77
  )
  set_tests_properties(
    pnmir_tensorrt_device_io
    PROPERTIES DEPENDS pnmir_tensorrt_integration
  )

  if(TARGET pnmir_persistent_device_state_test)
    add_test(
      NAME pnmir_tensorrt_persistent_device_state
      COMMAND
        pnmir_persistent_device_state_test
        tensorrt
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt-affine.pnmir
    )
    set_tests_properties(
      pnmir_tensorrt_persistent_device_state
      PROPERTIES
        DEPENDS pnmir_tensorrt_integration
        SKIP_RETURN_CODE 77
    )
  endif()
endif()

# Development integrations retain the migrated tiny model factories, while
# using the one exporter implementation in the sibling builder distribution.
get_property(pnmir_tests DIRECTORY PROPERTY TESTS)
foreach(pnmir_test IN LISTS pnmir_tests)
  set_property(TEST ${pnmir_test} APPEND PROPERTY ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../model-builder/src"
  )
endforeach()
