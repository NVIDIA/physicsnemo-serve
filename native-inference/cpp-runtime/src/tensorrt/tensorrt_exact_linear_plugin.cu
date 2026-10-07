#include "physicsnemo/inference/backends/tensorrt_exact.hpp"
#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactLinear";
constexpr char kPluginVersion[] = "1";
constexpr std::size_t kWorkspaceBytes = 1024U * 1024U;

__global__ void copy_linear_bias(const float* bias, float* output,
                                std::int64_t elements, int columns) {
  for (auto index = static_cast<std::int64_t>(blockDim.x) * blockIdx.x +
                    threadIdx.x;
       index < elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    output[index] = bias[index % columns];
  }
}

bool configured_workspace_size(const char* config, std::size_t* bytes) {
  if (config == nullptr) return true;
  std::size_t total = 0;
  bool matched = false;
  // Match Torch's :SIZE_KIB:COUNT pairs, including multiple workspace pools.
  for (const char* cursor = config; *cursor != '\0';) {
    if (*cursor != ':' || cursor[1] < '0' || cursor[1] > '9') {
      ++cursor;
      continue;
    }
    char* size_end = nullptr;
    errno = 0;
    const auto size = std::strtoull(cursor + 1, &size_end, 10);
    const bool size_overflow = errno == ERANGE;
    if (*size_end != ':' || size_end[1] < '0' || size_end[1] > '9') {
      ++cursor;
      continue;
    }
    char* count_end = nullptr;
    errno = 0;
    const auto count = std::strtoull(size_end + 1, &count_end, 10);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (size_overflow || errno == ERANGE || size > limit / 1024U ||
        (size != 0 && count > (limit - total) / (size * 1024U))) {
      return false;
    }
    total += static_cast<std::size_t>(size * count * 1024U);
    matched = true;
    cursor = count_end;
  }
  if (matched) {
    *bytes = total;
  } else {
    std::fprintf(stderr,
                 "Ignoring invalid CUBLAS_WORKSPACE_CONFIG for exact linear\n");
  }
  return true;
}

bool gemm_workspace_size(std::size_t* bytes) {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return false;
  }
  // PyTorch 2.10 CublasHandlePool.cpp: Hopper uses 32 MiB, other CUDA
  // devices use :4096:2:16:8. This is distinct from the Lt 1 MiB workspace.
  *bytes = properties.major == 9 && properties.minor == 0
               ? 32U * 1024U * 1024U
               : 4096U * 1024U * 2U + 16U * 1024U * 8U;
#if defined(_MSC_VER)
  char* config = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&config, &length, "CUBLAS_WORKSPACE_CONFIG") != 0) return false;
  const bool valid = configured_workspace_size(config, bytes);
  std::free(config);
  return valid;
#else
  return configured_workspace_size(std::getenv("CUBLAS_WORKSPACE_CONFIG"), bytes);
#endif
}

bool valid_dimensions(const nvinfer1::Dims& input,
                      const nvinfer1::Dims& weight,
                      const nvinfer1::Dims& bias) {
  return input.nbDims >= 2 && weight.nbDims == 2 && bias.nbDims == 1 &&
         input.d[input.nbDims - 1] > 0 && weight.d[0] > 0 &&
         weight.d[1] > 0 && input.d[input.nbDims - 1] == weight.d[1] &&
         bias.d[0] == weight.d[0];
}

bool matrix_rows(const nvinfer1::Dims& input, std::int32_t* rows) {
  std::int64_t result = 1;
  for (int32_t index = 0; index + 1 < input.nbDims; ++index) {
    if (input.d[index] <= 0) return false;
    result *= input.d[index];
    if (result > std::numeric_limits<std::int32_t>::max()) return false;
  }
  *rows = static_cast<std::int32_t>(result);
  return true;
}

std::uint32_t pointer_alignment(const void* pointer) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  for (std::uint32_t alignment = 256; alignment > 1; alignment /= 2) {
    if (address % alignment == 0) return alignment;
  }
  return 1;
}

struct Descriptors {
  ~Descriptors() {
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
  }

  cublasLtMatmulDesc_t operation{nullptr};
  cublasLtMatrixLayout_t a{nullptr};
  cublasLtMatrixLayout_t b{nullptr};
  cublasLtMatrixLayout_t c{nullptr};
  cublasLtMatmulPreference_t preference{nullptr};
};

class ExactLinearPlugin final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
 public:
  ~ExactLinearPlugin() override {
    if (workspace_ != nullptr) cudaFree(workspace_);
    if (handle_ != nullptr) cublasLtDestroy(handle_);
    if (gemm_workspace_ != nullptr) cudaFree(gemm_workspace_);
    if (gemm_handle_ != nullptr) cublasDestroy(gemm_handle_);
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactLinearPlugin();
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override { return ""; }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t nb_inputs,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t nb_outputs) noexcept override {
    if (inputs == nullptr || nb_inputs != 3 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].desc.dims, inputs[1].desc.dims,
                            inputs[2].desc.dims)
               ? 0
               : 1;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_inputs != 3 ||
        nb_outputs != 1) {
      return 1;
    }
    for (int32_t index = 0; index < nb_inputs; ++index) {
      if (input_types[index] != nvinfer1::DataType::kFLOAT) return 1;
    }
    output_types[0] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                          int32_t nb_inputs, const nvinfer1::DimsExprs*,
                          int32_t nb_shape_inputs,
                          nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                          nvinfer1::IExprBuilder&) noexcept override {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 ||
        nb_shape_inputs != 0 || nb_outputs != 1 || inputs[0].nbDims < 2 ||
        inputs[1].nbDims != 2) {
      return 1;
    }
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] = inputs[1].d[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    return tensorrt_detail::supports_fp32_linear(
        position, in_out, nb_inputs, nb_outputs, 3);
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t nb_inputs, const nvinfer1::PluginTensorDesc*,
                        int32_t nb_outputs) noexcept override {
    if (inputs == nullptr || nb_inputs != 3 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].dims, inputs[1].dims, inputs[2].dims)
               ? 0
               : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims, input_desc[1].dims,
                          input_desc[2].dims)) {
      return 1;
    }
    std::int32_t rows = 0;
    if (!matrix_rows(input_desc[0].dims, &rows)) return 1;
    const auto columns = input_desc[1].dims.d[0];
    const auto inner = input_desc[1].dims.d[1];
    // Torch addmm excludes every dimension-one matrix from the Lt path.
    if (rows == 1 || columns == 1 || inner == 1) {
      if (columns > std::numeric_limits<int>::max() ||
          inner > std::numeric_limits<int>::max()) {
        return 1;
      }
      return enqueue_gemm(rows, static_cast<int>(columns),
                          static_cast<int>(inner), inputs, outputs[0], stream);
    }
    if (handle_ == nullptr && cublasLtCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
      return 1;
    }
    if (workspace_ == nullptr &&
        cudaMalloc(&workspace_, kWorkspaceBytes) != cudaSuccess) {
      return 1;
    }

    Descriptors descriptors;
    if (cublasLtMatmulDescCreate(&descriptors.operation,
                                 CUBLAS_COMPUTE_32F,
                                 CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&descriptors.a, CUDA_R_32F, inner, columns,
                                   inner) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&descriptors.b, CUDA_R_32F, inner, rows,
                                   inner) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&descriptors.c, CUDA_R_32F, columns, rows,
                                   columns) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceCreate(&descriptors.preference) !=
            CUBLAS_STATUS_SUCCESS) {
      return 1;
    }

    constexpr cublasOperation_t transpose_a = CUBLAS_OP_T;
    constexpr cublasOperation_t transpose_b = CUBLAS_OP_N;
    constexpr cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const void* bias = inputs[2];
    if (cublasLtMatmulDescSetAttribute(
            descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
            sizeof(transpose_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
            sizeof(transpose_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
            sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(
            descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
            sizeof(bias)) != CUBLAS_STATUS_SUCCESS) {
      return 1;
    }

    const auto alignment_a = pointer_alignment(inputs[1]);
    const auto alignment_b = pointer_alignment(inputs[0]);
    const auto alignment_c = pointer_alignment(outputs[0]);
    const auto alignment_d = pointer_alignment(outputs[0]);
    if (cublasLtMatmulPreferenceSetAttribute(
            descriptors.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            descriptors.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            &alignment_a, sizeof(alignment_a)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            descriptors.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            &alignment_b, sizeof(alignment_b)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            descriptors.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            &alignment_c, sizeof(alignment_c)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(
            descriptors.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            &alignment_d, sizeof(alignment_d)) != CUBLAS_STATUS_SUCCESS) {
      return 1;
    }

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (cublasLtMatmulAlgoGetHeuristic(
            handle_, descriptors.operation, descriptors.a, descriptors.b,
            descriptors.c, descriptors.c, descriptors.preference, 1,
            &heuristic, &returned) != CUBLAS_STATUS_SUCCESS ||
        returned == 0) {
      return 1;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto status = cublasLtMatmul(
        handle_, descriptors.operation, &alpha, inputs[1], descriptors.a,
        inputs[0], descriptors.b, &beta, outputs[0], descriptors.c, outputs[0],
        descriptors.c, &heuristic.algo, workspace_, kWorkspaceBytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "TensorRT exact linear cuBLASLt call failed: %d\n",
                   static_cast<int>(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize()
      noexcept override {
    return &fields_;
  }

 private:
  int32_t enqueue_gemm(int rows, int columns, int inner,
                      const void* const* inputs, void* output,
                      cudaStream_t stream) noexcept {
    if (gemm_handle_ == nullptr &&
        cublasCreate(&gemm_handle_) != CUBLAS_STATUS_SUCCESS) {
      return 1;
    }
    if (!gemm_workspace_ready_) {
      if (!gemm_workspace_size(&gemm_workspace_bytes_) ||
          (gemm_workspace_bytes_ != 0 &&
           cudaMalloc(&gemm_workspace_, gemm_workspace_bytes_) != cudaSuccess)) {
        return 1;
      }
      gemm_workspace_ready_ = true;
    }
    // Setting the stream resets cuBLAS workspace, so bind it afterwards.
    if (cublasSetStream(gemm_handle_, stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(gemm_handle_, gemm_workspace_, gemm_workspace_bytes_) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(gemm_handle_, CUBLAS_DEFAULT_MATH) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasSetPointerMode(gemm_handle_, CUBLAS_POINTER_MODE_HOST) !=
            CUBLAS_STATUS_SUCCESS) {
      return 1;
    }
    // addmm broadcasts bias into C before GEMM with beta=1. Copying preserves
    // the original arithmetic, including the single-product (inner=1) case.
    const auto elements = static_cast<std::int64_t>(rows) * columns;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        std::min<std::int64_t>((elements + threads - 1) / threads, 65535));
    copy_linear_bias<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(inputs[2]), static_cast<float*>(output),
        elements, columns);
    if (cudaGetLastError() != cudaSuccess) return 1;

    // weight.t() is contiguous when either weight dimension is one. These
    // flags/leading dimensions match cuBlasCommonArgs and its singleton fixup.
    const auto transpose_a =
        columns == 1 || inner == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int leading_a = transpose_a == CUBLAS_OP_N ? columns : inner;
    constexpr float alpha = 1.0F;
    constexpr float beta = 1.0F;
    const auto status = cublasSgemm(
        gemm_handle_, transpose_a, CUBLAS_OP_N, columns, rows, inner, &alpha,
        static_cast<const float*>(inputs[1]), leading_a,
        static_cast<const float*>(inputs[0]), inner, &beta,
        static_cast<float*>(output), columns);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "TensorRT exact linear cuBLAS call failed: %d\n",
                   static_cast<int>(status));
      return 1;
    }
    return 0;
  }

  cublasLtHandle_t handle_{nullptr};
  void* workspace_{nullptr};
  cublasHandle_t gemm_handle_{nullptr};
  void* gemm_workspace_{nullptr};
  std::size_t gemm_workspace_bytes_{0};
  bool gemm_workspace_ready_{false};
  nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

class ExactLinearPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactLinearPlugin();
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &fields_;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override { return ""; }

 private:
  nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

REGISTER_TENSORRT_PLUGIN(ExactLinearPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_linear_register() noexcept { return 0; }
