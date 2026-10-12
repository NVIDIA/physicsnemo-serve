#include "physicsnemo/inference/backends/tensorrt_exact.hpp"

#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactScalarDiv";
constexpr char kPluginVersion[] = "1";
constexpr int kThreads = 256;

__global__ void exact_scalar_div_kernel(const float* input,
                                        const float* scalar, float* output,
                                        std::int64_t elements) {
  const float reciprocal = 1.0F / scalar[0];
  for (auto index = static_cast<std::int64_t>(blockDim.x) * blockIdx.x +
                    threadIdx.x;
       index < elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    output[index] = input[index] * reciprocal;
  }
}

using tensorrt_detail::element_count;

bool valid_dimensions(const nvinfer1::Dims& input,
                      const nvinfer1::Dims& scalar) {
  std::int64_t elements = 0;
  return element_count(input, &elements) && scalar.nbDims == 0;
}

class ExactScalarDivPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime {
 public:
  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactScalarDivPlugin();
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
    return inputs != nullptr && nb_inputs == 2 && nb_outputs == 1 &&
                   valid_dimensions(inputs[0].desc.dims, inputs[1].desc.dims)
               ? 0
               : 1;
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
        nb_shape_inputs != 0 || nb_outputs != 1 || inputs[1].nbDims != 0) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    return tensorrt_detail::supports_fp32_linear(
        position, in_out, nb_inputs, nb_outputs, 2);
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t nb_inputs, const nvinfer1::PluginTensorDesc*,
                        int32_t nb_outputs) noexcept override {
    return inputs != nullptr && nb_inputs == 2 && nb_outputs == 1 &&
                   valid_dimensions(inputs[0].dims, inputs[1].dims)
               ? 0
               : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims, input_desc[1].dims)) {
      return 1;
    }
    std::int64_t elements = 0;
    if (!element_count(input_desc[0].dims, &elements)) return 1;
    const auto blocks_64 = (elements + kThreads - 1) / kThreads;
    const auto blocks = static_cast<int>(blocks_64 > 65535 ? 65535 : blocks_64);
    exact_scalar_div_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<float*>(outputs[0]), elements);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact scalar Div launch failed: %s\n",
                   cudaGetErrorString(status));
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
  nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

class ExactScalarDivPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactScalarDivPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactScalarDivPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_scalar_div_register() noexcept {
  return 0;
}
