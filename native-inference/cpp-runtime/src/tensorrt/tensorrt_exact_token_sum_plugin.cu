#include "physicsnemo/inference/backends/tensorrt_exact.hpp"
#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactTokenSum";
constexpr char kPluginVersion[] = "1";
constexpr int kVectorSize = 4;
constexpr int kMaxThreads = 512 / kVectorSize;
constexpr int kWarpSize = 32;
constexpr int kValuesPerThread = 4;

struct ReductionConfig {
  int outputs;
  int tokens;
  int token_stride;
  int block_width;
  int block_height;
  int output_step;
  int input_step;
  int ctas_per_output;
  int grid_x;
  bool split_across_warps;
};

int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

int last_power_of_two(int value) {
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return std::max(1, value - (value >> 1));
}

bool valid_dimensions(const nvinfer1::Dims& input) {
  if (input.nbDims != 4 || input.d[0] != 1) return false;
  for (int32_t index = 0; index < input.nbDims; ++index) {
    if (input.d[index] <= 0 ||
        input.d[index] > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  const auto outputs = static_cast<std::int64_t>(input.d[0]) * input.d[2] *
                       input.d[3];
  const auto token_stride = input.d[2] * input.d[3];
  return outputs <= std::numeric_limits<int>::max() &&
         token_stride <= std::numeric_limits<int>::max() &&
         outputs % kVectorSize == 0;
}

ReductionConfig make_config(const nvinfer1::Dims& input) {
  const int outputs =
      static_cast<int>(input.d[0] * input.d[2] * input.d[3]);
  const int tokens = static_cast<int>(input.d[1]);
  const int output_vectors = outputs / kVectorSize;
  int dim0_power = output_vectors < kMaxThreads
                       ? last_power_of_two(output_vectors)
                       : kMaxThreads;
  int dim1_power = tokens < kMaxThreads
                       ? last_power_of_two(tokens)
                       : kMaxThreads;
  int block_width = std::min(dim0_power, kWarpSize);
  int block_height = std::min(dim1_power, kMaxThreads / block_width);
  block_width = std::min(dim0_power, kMaxThreads / block_height);

  const int threshold = std::min(block_height * 16, 256);
  const bool split_across_warps = tokens >= threshold;
  int output_step = block_width;
  int input_step = 1;
  if (split_across_warps) {
    input_step *= block_height;
  } else {
    output_step *= block_height;
  }

  int ctas_per_output = 1;
  const int values_per_thread = ceil_div(tokens, input_step);
  int device = 0;
  cudaDeviceProp properties{};
  if (split_across_warps && values_per_thread >= 256 &&
      cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    const int grid_x = ceil_div(output_vectors, output_step);
    const int blocks_per_sm =
        properties.maxThreadsPerMultiProcessor /
        (block_width * block_height);
    const int target_grid = properties.multiProcessorCount * blocks_per_sm;
    if (grid_x <= target_grid) {
      const int by_occupancy = ceil_div(target_grid, grid_x);
      const int by_minimum_work = ceil_div(values_per_thread, 16);
      const int by_maximum_work = ceil_div(values_per_thread, 256);
      ctas_per_output = std::max(
          std::min(by_occupancy, by_minimum_work), by_maximum_work);
      if (ctas_per_output > 1) input_step *= ctas_per_output;
    }
  }

  return {outputs,
          tokens,
          static_cast<int>(input.d[2] * input.d[3]),
          block_width,
          block_height,
          output_step,
          input_step,
          ctas_per_output,
          ceil_div(output_vectors, output_step),
          split_across_warps};
}

std::size_t workspace_size(const ReductionConfig& config) {
  if (config.ctas_per_output == 1) return 0;
  const std::size_t staged_vectors =
      static_cast<std::size_t>(config.grid_x) * config.ctas_per_output *
      config.block_width;
  const std::size_t staging = staged_vectors * sizeof(float4);
  constexpr std::size_t alignment = 256;
  const std::size_t aligned_staging =
      (staging + alignment - 1) & ~(alignment - 1);
  return aligned_staging +
         static_cast<std::size_t>(config.grid_x) * sizeof(int);
}

__device__ float4 add_vectors(const float4& left, const float4& right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z,
          left.w + right.w};
}

__device__ float4 thread_reduce(const float4* input, int output_vector,
                                int token_start, int token_step, int tokens,
                                int token_stride_vectors) {
  float4 accumulators[kValuesPerThread]{};
  int token = token_start;
  while (token + (kValuesPerThread - 1) * token_step < tokens) {
#pragma unroll
    for (int index = 0; index < kValuesPerThread; ++index) {
      const float4 value =
          input[(token + index * token_step) * token_stride_vectors +
                output_vector];
      accumulators[index] = add_vectors(accumulators[index], value);
    }
    token += token_step * kValuesPerThread;
  }
#pragma unroll
  for (int index = 0; index < kValuesPerThread; ++index) {
    if (token >= tokens) break;
    const float4 value =
        input[token * token_stride_vectors + output_vector];
    accumulators[index] = add_vectors(accumulators[index], value);
    token += token_step;
  }
#pragma unroll
  for (int index = 1; index < kValuesPerThread; ++index) {
    accumulators[0] = add_vectors(accumulators[0], accumulators[index]);
  }
  return accumulators[0];
}

__device__ float4 block_y_reduce(float4 value, float4* shared) {
  const int offset = threadIdx.x + threadIdx.y * blockDim.x;
  shared[offset] = value;
  for (int distance = blockDim.y / 2; distance > 0; distance >>= 1) {
    __syncthreads();
    if (threadIdx.y < distance && threadIdx.y + distance < blockDim.y) {
      value = add_vectors(value, shared[offset + distance * blockDim.x]);
      shared[offset] = value;
    }
  }
  return value;
}

__global__ void exact_token_sum_kernel(ReductionConfig config,
                                       const float* raw_input,
                                       float* raw_output, float4* staging,
                                       int* semaphores) {
  extern __shared__ float4 shared[];
  const auto* input = reinterpret_cast<const float4*>(raw_input);
  auto* output = reinterpret_cast<float4*>(raw_output);
  const int output_vector =
      config.split_across_warps
          ? blockIdx.x * config.output_step + threadIdx.x
          : blockIdx.x * config.output_step +
                threadIdx.y * config.block_width + threadIdx.x;
  const int output_vectors = config.outputs / kVectorSize;
  float4 value{};
  if (output_vector < output_vectors) {
    const int token_start =
        config.split_across_warps
            ? threadIdx.y + blockIdx.y * config.block_height
            : 0;
    if (token_start < config.tokens) {
      value = thread_reduce(input, output_vector, token_start,
                            config.input_step, config.tokens,
                            config.token_stride / kVectorSize);
    }
  }

  if (config.split_across_warps) value = block_y_reduce(value, shared);

  if (config.ctas_per_output == 1) {
    if ((!config.split_across_warps || threadIdx.y == 0) &&
        output_vector < output_vectors) {
      output[output_vector] = value;
    }
    return;
  }

  if (threadIdx.y == 0 && output_vector < output_vectors) {
    const int staging_offset =
        threadIdx.x +
        (blockIdx.y + blockIdx.x * gridDim.y) * config.block_width;
    staging[staging_offset] = value;
  }
  __threadfence();
  __syncthreads();

  __shared__ bool is_last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const int previous = atomicAdd(&semaphores[blockIdx.x], 1);
    is_last_block = previous == gridDim.y - 1;
  }
  __syncthreads();
  if (!is_last_block) return;
  __threadfence();

  value = {};
  if (output_vector < output_vectors) {
    for (int cta = threadIdx.y; cta < config.ctas_per_output;
         cta += blockDim.y) {
      const int staging_offset =
          threadIdx.x +
          (cta + blockIdx.x * gridDim.y) * config.block_width;
      value = add_vectors(value, staging[staging_offset]);
    }
  }
  value = block_y_reduce(value, shared);
  if (threadIdx.y == 0 && output_vector < output_vectors) {
    output[output_vector] = value;
  }
}

class ExactTokenSumPlugin final : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
 public:
  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactTokenSumPlugin();
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
    if (inputs == nullptr || nb_inputs != 1 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].desc.dims) ? 0 : 1;
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
        nb_shape_inputs != 0 || nb_outputs != 1 || inputs[0].nbDims != 4) {
      return 1;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[2];
    outputs[0].d[2] = inputs[0].d[3];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    return tensorrt_detail::supports_fp32_linear(
        position, in_out, nb_inputs, nb_outputs, 1);
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  std::size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nb_inputs,
      const nvinfer1::DynamicPluginTensorDesc*,
      int32_t nb_outputs) const noexcept override {
    if (inputs == nullptr || nb_inputs != 1 || nb_outputs != 1 ||
        !valid_dimensions(inputs[0].desc.dims)) {
      return 0;
    }
    return workspace_size(make_config(inputs[0].desc.dims));
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t nb_inputs, const nvinfer1::PluginTensorDesc*,
                        int32_t nb_outputs) noexcept override {
    if (inputs == nullptr || nb_inputs != 1 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].dims) ? 0 : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims)) {
      return 1;
    }
    const ReductionConfig config = make_config(input_desc[0].dims);
    float4* staging = nullptr;
    int* semaphores = nullptr;
    if (config.ctas_per_output > 1) {
      if (workspace == nullptr) return 1;
      staging = static_cast<float4*>(workspace);
      const std::size_t staging_bytes =
          static_cast<std::size_t>(config.grid_x) * config.ctas_per_output *
          config.block_width * sizeof(float4);
      constexpr std::size_t alignment = 256;
      const std::size_t aligned_staging =
          (staging_bytes + alignment - 1) & ~(alignment - 1);
      semaphores = reinterpret_cast<int*>(
          static_cast<std::byte*>(workspace) + aligned_staging);
      const auto status = cudaMemsetAsync(
          semaphores, 0, static_cast<std::size_t>(config.grid_x) * sizeof(int),
          stream);
      if (status != cudaSuccess) return 1;
    }

    const dim3 block(config.block_width, config.block_height);
    const dim3 grid(config.grid_x, config.ctas_per_output);
    const auto shared_bytes = static_cast<std::size_t>(config.block_width) *
                              config.block_height * sizeof(float4);
    exact_token_sum_kernel<<<grid, block, shared_bytes, stream>>>(
        config, static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]), staging, semaphores);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact token sum launch failed: %s\n",
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

class ExactTokenSumPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactTokenSumPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactTokenSumPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_token_sum_register() noexcept { return 0; }
