#include "physicsnemo/inference/backends/tensorrt_exact_weighted_blend_plugin.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactWeightedBlend";
constexpr char kPluginVersion[] = "1";
constexpr int kThreads = 256;

__global__ void exact_weighted_blend_kernel(
    const float* left, const float* left_weight, const float* right,
    const float* right_weight, float* output, std::int64_t elements) {
  for (auto index = static_cast<std::int64_t>(blockDim.x) * blockIdx.x +
                    threadIdx.x;
       index < elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const float scaled_left = __fmul_rn(left[index], left_weight[0]);
    const float scaled_right = __fmul_rn(right[index], right_weight[0]);
    output[index] = __fadd_rn(scaled_left, scaled_right);
  }
}

bool element_count(const nvinfer1::Dims& input, std::int64_t* elements) {
  if (input.nbDims < 1) return false;
  std::int64_t result = 1;
  for (int32_t index = 0; index < input.nbDims; ++index) {
    if (input.d[index] <= 0 ||
        result > std::numeric_limits<std::int64_t>::max() / input.d[index]) {
      return false;
    }
    result *= input.d[index];
  }
  *elements = result;
  return true;
}

bool same_dimensions(const nvinfer1::Dims& left,
                     const nvinfer1::Dims& right) {
  if (left.nbDims != right.nbDims) return false;
  for (int32_t index = 0; index < left.nbDims; ++index) {
    if (left.d[index] != right.d[index]) return false;
  }
  return true;
}

bool valid_dimensions(const nvinfer1::Dims& left,
                      const nvinfer1::Dims& left_weight,
                      const nvinfer1::Dims& right,
                      const nvinfer1::Dims& right_weight) {
  std::int64_t elements = 0;
  return element_count(left, &elements) && same_dimensions(left, right) &&
         left_weight.nbDims == 0 && right_weight.nbDims == 0;
}

class ExactWeightedBlendPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
 public:
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
    return new (std::nothrow) ExactWeightedBlendPlugin();
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
    return inputs != nullptr && nb_inputs == 4 && nb_outputs == 1 &&
                   valid_dimensions(
                       inputs[0].desc.dims, inputs[1].desc.dims,
                       inputs[2].desc.dims, inputs[3].desc.dims)
               ? 0
               : 1;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_inputs != 4 ||
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
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 4 ||
        nb_shape_inputs != 0 || nb_outputs != 1 ||
        inputs[1].nbDims != 0 || inputs[3].nbDims != 0) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (in_out == nullptr || position < 0 || position >= 5 || nb_inputs != 4 ||
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
    return inputs != nullptr && nb_inputs == 4 && nb_outputs == 1 &&
                   valid_dimensions(inputs[0].dims, inputs[1].dims,
                                    inputs[2].dims, inputs[3].dims)
               ? 0
               : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims, input_desc[1].dims,
                          input_desc[2].dims, input_desc[3].dims)) {
      return 1;
    }
    std::int64_t elements = 0;
    if (!element_count(input_desc[0].dims, &elements)) return 1;
    const auto blocks_64 = (elements + kThreads - 1) / kThreads;
    const auto blocks = static_cast<int>(blocks_64 > 65535 ? 65535 : blocks_64);
    exact_weighted_blend_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        static_cast<float*>(outputs[0]), elements);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "TensorRT exact weighted-blend launch failed: %s\n",
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

class ExactWeightedBlendPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactWeightedBlendPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactWeightedBlendPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_weighted_blend_register() noexcept {
  return 0;
}
