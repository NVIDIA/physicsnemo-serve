#include "physicsnemo/inference/backends/tensorrt_exact.hpp"
#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactInverseDistanceBlend";
constexpr char kPluginVersion[] = "1";
constexpr int kThreads = 256;

__global__ void exact_inverse_distance_blend_kernel(
    const float* const* inputs, float* output, std::int64_t elements,
    int samples) {
  for (auto index = static_cast<std::int64_t>(blockDim.x) * blockIdx.x +
                    threadIdx.x;
       index < elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    float neighbor_sum = 0.0F;
    float inverse_distance_sum = 0.0F;
    for (int sample = 0; sample < samples; ++sample) {
      const float prediction = inputs[1 + 2 * sample][index];
      const float distance = inputs[2 + 2 * sample][index];
      const float inverse_distance = __fdiv_rn(1.0F, distance);
      const float weighted = __fmul_rn(prediction, inverse_distance);
      if (sample == 0) {
        neighbor_sum = weighted;
        inverse_distance_sum = inverse_distance;
      } else {
        neighbor_sum = __fadd_rn(neighbor_sum, weighted);
        inverse_distance_sum =
            __fadd_rn(inverse_distance_sum, inverse_distance);
      }
    }
    const float center_half = __fmul_rn(inputs[0][index], 0.5F);
    const float neighbor_half = __fmul_rn(neighbor_sum, 0.5F);
    const float normalized =
        __fdiv_rn(neighbor_half, inverse_distance_sum);
    output[index] = __fadd_rn(center_half, normalized);
  }
}

using tensorrt_detail::element_count;

bool same_dimensions(const nvinfer1::Dims& left,
                     const nvinfer1::Dims& right) {
  if (left.nbDims != right.nbDims) return false;
  for (int32_t index = 0; index < left.nbDims; ++index) {
    if (left.d[index] != right.d[index]) return false;
  }
  return true;
}

template <typename Descriptor>
bool valid_dimensions(const Descriptor* inputs, int32_t nb_inputs) {
  if (inputs == nullptr || nb_inputs < 3 || nb_inputs % 2 == 0) return false;
  std::int64_t elements = 0;
  if (!element_count(inputs[0].dims, &elements)) return false;
  for (int32_t index = 1; index < nb_inputs; ++index) {
    if (!same_dimensions(inputs[0].dims, inputs[index].dims)) return false;
  }
  return true;
}

class ExactInverseDistanceBlendPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime {
 public:
  ~ExactInverseDistanceBlendPlugin() override {
    if (device_inputs_ != nullptr) cudaFree(device_inputs_);
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactInverseDistanceBlendPlugin();
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
    if (inputs == nullptr || nb_outputs != 1) return 1;
    for (int32_t index = 0; index < nb_inputs; ++index) {
      if (inputs[index].desc.dims.nbDims != inputs[0].desc.dims.nbDims) return 1;
    }
    if (nb_inputs < 3 || nb_inputs % 2 == 0) return 1;
    input_count_ = nb_inputs;
    return 0;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t nb_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t nb_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
        nb_inputs < 3 || nb_inputs % 2 == 0) {
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
    if (inputs == nullptr || outputs == nullptr || nb_inputs < 3 ||
        nb_inputs % 2 == 0 || nb_shape_inputs != 0 || nb_outputs != 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (in_out == nullptr || position < 0 || position >= nb_inputs + 1 ||
        nb_inputs < 3 || nb_inputs % 2 == 0 || nb_outputs != 1) {
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
    if (nb_outputs != 1 || !valid_dimensions(inputs, nb_inputs)) return 1;
    input_count_ = nb_inputs;
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc, input_count_)) {
      return 1;
    }
    if (device_input_capacity_ < input_count_) {
      if (device_inputs_ != nullptr && cudaFree(device_inputs_) != cudaSuccess) {
        return 1;
      }
      if (cudaMalloc(&device_inputs_,
                     sizeof(void*) * static_cast<std::size_t>(input_count_)) !=
          cudaSuccess) {
        device_inputs_ = nullptr;
        device_input_capacity_ = 0;
        return 1;
      }
      device_input_capacity_ = input_count_;
    }
    if (cudaMemcpyAsync(device_inputs_, inputs,
                        sizeof(void*) * static_cast<std::size_t>(input_count_),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return 1;
    }
    std::int64_t elements = 0;
    if (!element_count(input_desc[0].dims, &elements)) return 1;
    const auto blocks_64 = (elements + kThreads - 1) / kThreads;
    const auto blocks = static_cast<int>(blocks_64 > 65535 ? 65535 : blocks_64);
    exact_inverse_distance_blend_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float* const*>(device_inputs_),
        static_cast<float*>(outputs[0]), elements, (input_count_ - 1) / 2);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "TensorRT exact inverse-distance blend launch failed: %s\n",
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
  int32_t input_count_{0};
  int32_t device_input_capacity_{0};
  void* device_inputs_{nullptr};
  nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

class ExactInverseDistanceBlendPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactInverseDistanceBlendPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactInverseDistanceBlendPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int
pnmir_tensorrt_exact_inverse_distance_blend_register() noexcept {
  return 0;
}
