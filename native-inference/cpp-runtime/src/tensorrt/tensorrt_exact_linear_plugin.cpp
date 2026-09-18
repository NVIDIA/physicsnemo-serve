#include "physicsnemo/inference/backends/tensorrt_exact_linear_plugin.hpp"

#include <NvInfer.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactLinear";
constexpr char kPluginVersion[] = "1";
constexpr std::size_t kWorkspaceBytes = 1024U * 1024U;

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
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
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
    if (in_out == nullptr || position < 0 || position >= 4 || nb_inputs != 3 ||
        nb_outputs != 1) {
      return false;
    }
    const auto& descriptor = in_out[position].desc;
    return descriptor.type == nvinfer1::DataType::kFLOAT &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
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
  cublasLtHandle_t handle_{nullptr};
  void* workspace_{nullptr};
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
