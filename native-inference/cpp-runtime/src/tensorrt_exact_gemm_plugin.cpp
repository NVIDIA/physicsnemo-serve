#include "physicsnemo/inference/backends/tensorrt_exact_gemm_plugin.hpp"

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactGemm";
constexpr char kPluginVersion[] = "1";

bool valid_dimensions(const nvinfer1::Dims& input,
                      const nvinfer1::Dims& weight) {
  return input.nbDims >= 2 && weight.nbDims == 2 &&
         input.d[input.nbDims - 1] > 0 && weight.d[0] > 0 &&
         weight.d[1] > 0 && input.d[input.nbDims - 1] == weight.d[1];
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

class ExactGemmPlugin final : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
 public:
  ExactGemmPlugin() {
    serialization_collection_.nbFields = 0;
    serialization_collection_.fields = nullptr;
  }

  ~ExactGemmPlugin() override {
    if (handle_ != nullptr) cublasDestroy(handle_);
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
    return new (std::nothrow) ExactGemmPlugin();
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
    if (inputs == nullptr || nb_inputs != 2 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].desc.dims, inputs[1].desc.dims) ? 0 : 1;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_inputs != 2 ||
        nb_outputs != 1 || input_types[0] != nvinfer1::DataType::kFLOAT ||
        input_types[1] != nvinfer1::DataType::kFLOAT) {
      return 1;
    }
    output_types[0] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                          int32_t nb_inputs, const nvinfer1::DimsExprs*,
                          int32_t nb_shape_inputs,
                          nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                          nvinfer1::IExprBuilder&) noexcept override {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 2 ||
        nb_shape_inputs != 0 || nb_outputs != 1 ||
        inputs[0].nbDims < 2 || inputs[1].nbDims != 2) {
      return 1;
    }
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] = inputs[1].d[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (in_out == nullptr || position < 0 || position >= 3 || nb_inputs != 2 ||
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
    if (inputs == nullptr || nb_inputs != 2 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].dims, inputs[1].dims) ? 0 : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims, input_desc[1].dims)) {
      return 1;
    }
    std::int32_t rows = 0;
    if (!matrix_rows(input_desc[0].dims, &rows)) return 1;
    const auto columns = input_desc[1].dims.d[0];
    const auto inner = input_desc[1].dims.d[1];

    if (handle_ == nullptr && cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
      return 1;
    }
    if (cublasSetStream(handle_, stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH) !=
            CUBLAS_STATUS_SUCCESS) {
      return 1;
    }

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto status = cublasSgemm(
        handle_, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner, &alpha,
        static_cast<const float*>(inputs[1]), inner,
        static_cast<const float*>(inputs[0]), inner, &beta,
        static_cast<float*>(outputs[0]), columns);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "TensorRT exact GEMM cuBLAS call failed: %d\n",
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
    return &serialization_collection_;
  }

 private:
  cublasHandle_t handle_{nullptr};
  nvinfer1::PluginFieldCollection serialization_collection_{};
};

class ExactGemmPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  ExactGemmPluginCreator() {
    collection_.nbFields = 0;
    collection_.fields = nullptr;
  }

  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactGemmPlugin();
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &collection_;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override { return ""; }

 private:
  nvinfer1::PluginFieldCollection collection_{};
};

REGISTER_TENSORRT_PLUGIN(ExactGemmPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_gemm_register() noexcept {
  return 0;
}
