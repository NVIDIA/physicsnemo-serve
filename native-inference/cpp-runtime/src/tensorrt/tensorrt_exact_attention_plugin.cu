#include "physicsnemo/inference/backends/tensorrt_exact_attention_plugin.hpp"

#include <tuple>

#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactAttention";
constexpr char kPluginVersion[] = "1";
using AttentionKernel = PyTorchMemEffAttention::AttentionKernel<
    float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>;

__global__ void __launch_bounds__(AttentionKernel::kNumThreads,
                                  AttentionKernel::kMinBlocksPerSm)
    exact_attention_kernel(AttentionKernel::Params parameters) {
  if (!parameters.advance_to_block()) return;
  AttentionKernel::attention_kernel(parameters);
}

bool valid_dimensions(const nvinfer1::Dims& query,
                      const nvinfer1::Dims& key,
                      const nvinfer1::Dims& value) {
  if (query.nbDims != 4 || key.nbDims != 4 || value.nbDims != 4) return false;
  for (int32_t index = 0; index < 4; ++index) {
    if (query.d[index] != key.d[index] || query.d[index] != value.d[index]) {
      return false;
    }
  }
  return query.d[0] > 0 && query.d[0] <= std::numeric_limits<int32_t>::max() &&
         query.d[1] > 0 && query.d[2] > 0 && query.d[3] > 0 &&
         query.d[3] <= AttentionKernel::kMaxK && query.d[3] % 4 == 0;
}

AttentionKernel::Params make_parameters(const void* const* inputs,
                                        void* output,
                                        const nvinfer1::Dims& dimensions) {
  const int batches = dimensions.d[0];
  const int heads = dimensions.d[1];
  const int sequence = dimensions.d[2];
  const int head_dimension = dimensions.d[3];
  AttentionKernel::Params parameters;
  parameters.query_ptr = static_cast<const float*>(inputs[0]);
  parameters.key_ptr = static_cast<const float*>(inputs[1]);
  parameters.value_ptr = static_cast<const float*>(inputs[2]);
  parameters.output_ptr = static_cast<float*>(output);
  parameters.num_batches = batches;
  parameters.num_heads = heads;
  parameters.num_queries = sequence;
  parameters.num_keys = sequence;
  parameters.head_dim = head_dimension;
  parameters.head_dim_value = head_dimension;
  parameters.scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dimension)));

  const int kHeadStride = sequence * head_dimension;
  const int kBatchStride = heads * kHeadStride;
  parameters.q_strideB = kBatchStride;
  parameters.k_strideB = kBatchStride;
  parameters.v_strideB = kBatchStride;
  parameters.q_strideM = head_dimension;
  parameters.k_strideM = head_dimension;
  parameters.v_strideM = head_dimension;
  parameters.q_strideH = kHeadStride;
  parameters.k_strideH = kHeadStride;
  parameters.v_strideH = kHeadStride;
  parameters.o_strideM = heads * head_dimension;
  return parameters;
}

class ExactAttentionPlugin final : public nvinfer1::IPluginV3,
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
    return new (std::nothrow) ExactAttentionPlugin();
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
        nb_shape_inputs != 0 || nb_outputs != 1 || inputs[0].nbDims != 4) {
      return 1;
    }
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[2];
    outputs[0].d[2] = inputs[0].d[1];
    outputs[0].d[3] = inputs[0].d[3];
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
    const auto shared_bytes = sizeof(AttentionKernel::SharedStorage);
    if (shared_bytes > 0xC000U) {
      const auto attribute_status = cudaFuncSetAttribute(
          exact_attention_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes));
      if (attribute_status != cudaSuccess) return 1;
    }
    auto parameters = make_parameters(inputs, outputs[0], input_desc[0].dims);
    exact_attention_kernel<<<parameters.getBlocksGrid(),
                             parameters.getThreadsGrid(), shared_bytes,
                             stream>>>(parameters);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact attention launch failed: %s\n",
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

class ExactAttentionPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactAttentionPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactAttentionPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_attention_register() noexcept { return 0; }
