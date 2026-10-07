#include "physicsnemo/inference/backends/tensorrt_exact.hpp"

#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactGelu";
constexpr char kPluginVersion[] = "1";
constexpr int kThreads = 256;

__global__ void exact_gelu_kernel(const float* input, float* output,
                                  std::int64_t elements) {
  constexpr float kAlpha = 0.70710678118654752440;
  for (auto index = static_cast<std::int64_t>(blockDim.x) * blockIdx.x +
                    threadIdx.x;
       index < elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const float value = input[index];
    output[index] = value * 0.5F * (1.0F + ::erf(value * kAlpha));
  }
}

using tensorrt_detail::element_count;

class ExactGeluPlugin final : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
 public:
  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactGeluPlugin();
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
    std::int64_t elements = 0;
    return inputs != nullptr && nb_inputs == 1 && nb_outputs == 1 &&
                   element_count(inputs[0].desc.dims, &elements)
               ? 0
               : 1;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_inputs != 1 ||
        nb_outputs != 1 || input_types[0] != nvinfer1::DataType::kFLOAT) {
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
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 1 ||
        nb_shape_inputs != 0 || nb_outputs != 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    return tensorrt_detail::supports_fp32_linear(
        position, in_out, nb_inputs, nb_outputs, 1);
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t nb_inputs, const nvinfer1::PluginTensorDesc*,
                        int32_t nb_outputs) noexcept override {
    std::int64_t elements = 0;
    return inputs != nullptr && nb_inputs == 1 && nb_outputs == 1 &&
                   element_count(inputs[0].dims, &elements)
               ? 0
               : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    std::int64_t elements = 0;
    if (!element_count(input_desc[0].dims, &elements)) return 1;
    const auto blocks_64 = (elements + kThreads - 1) / kThreads;
    const auto blocks = static_cast<int>(
        blocks_64 > 65535 ? 65535 : blocks_64);
    exact_gelu_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]), elements);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact GELU launch failed: %s\n",
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

class ExactGeluPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactGeluPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactGeluPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_gelu_register() noexcept { return 0; }
