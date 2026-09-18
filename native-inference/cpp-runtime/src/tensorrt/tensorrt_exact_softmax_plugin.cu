#include "physicsnemo/inference/backends/tensorrt_exact_softmax_plugin.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactSoftmax";
constexpr char kPluginVersion[] = "1";
constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

template <typename Reduce>
__device__ __forceinline__ float warp_reduce(float value, Reduce reduce) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = reduce(value,
                   __shfl_xor_sync(0xFFFFFFFFU, value, offset, kWarpSize));
  }
  return value;
}

struct Add {
  __device__ __forceinline__ float operator()(float left,
                                               float right) const {
    return left + right;
  }
};

struct Max {
  __device__ __forceinline__ float operator()(float left,
                                               float right) const {
    return left < right ? right : left;
  }
};

template <int kElements, int kWarpBatch>
__global__ void exact_softmax_kernel(const float* input, float* output,
                                     int rows) {
  constexpr int kIterations = kElements / kWarpSize;
  const int first_row =
      (static_cast<int>(blockDim.y) * blockIdx.x + threadIdx.y) * kWarpBatch;
  int local_rows = rows - first_row;
  if (local_rows <= 0) return;
  if (local_rows > kWarpBatch) local_rows = kWarpBatch;

  const int lane = threadIdx.x;
  const auto offset = static_cast<std::int64_t>(first_row) * kElements + lane;
  const float* input_row = input + offset;
  float* output_row = output + offset;

  float elements[kWarpBatch][kIterations];
#pragma unroll
  for (int row = 0; row < kWarpBatch; ++row) {
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      elements[row][iteration] =
          row < local_rows
              ? input_row[row * kElements + iteration * kWarpSize]
              : __int_as_float(0xFF800000);
    }
  }

  float maximum[kWarpBatch];
#pragma unroll
  for (int row = 0; row < kWarpBatch; ++row) {
    maximum[row] = elements[row][0];
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      maximum[row] = maximum[row] > elements[row][iteration]
                         ? maximum[row]
                         : elements[row][iteration];
    }
    maximum[row] = warp_reduce(maximum[row], Max{});
  }

  float sum[kWarpBatch]{};
#pragma unroll
  for (int row = 0; row < kWarpBatch; ++row) {
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      elements[row][iteration] =
          std::exp(elements[row][iteration] - maximum[row]);
      sum[row] += elements[row][iteration];
    }
    sum[row] = warp_reduce(sum[row], Add{});
  }

#pragma unroll
  for (int row = 0; row < kWarpBatch; ++row) {
    if (row >= local_rows) break;
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      output_row[row * kElements + iteration * kWarpSize] =
          sum[row] == 0.0F ? __int_as_float(0x7FC00000)
                           : elements[row][iteration] / sum[row];
    }
  }
}

bool valid_dimensions(const nvinfer1::Dims& input) {
  if (input.nbDims < 1) return false;
  const int width = input.d[input.nbDims - 1];
  return width == 128 || width == 512;
}

bool row_count(const nvinfer1::Dims& input, std::int32_t* rows) {
  std::int64_t result = 1;
  for (int32_t index = 0; index + 1 < input.nbDims; ++index) {
    if (input.d[index] <= 0) return false;
    result *= input.d[index];
    if (result > std::numeric_limits<std::int32_t>::max()) return false;
  }
  *rows = static_cast<std::int32_t>(result);
  return true;
}

class ExactSoftmaxPlugin final : public nvinfer1::IPluginV3,
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
    return new (std::nothrow) ExactSoftmaxPlugin();
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
        nb_shape_inputs != 0 || nb_outputs != 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override {
    if (in_out == nullptr || position < 0 || position >= 2 || nb_inputs != 1 ||
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
    if (inputs == nullptr || nb_inputs != 1 || nb_outputs != 1) return 1;
    return valid_dimensions(inputs[0].dims) ? 0 : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc*,
                  const void* const* inputs, void* const* outputs, void*,
                  cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        !valid_dimensions(input_desc[0].dims)) {
      return 1;
    }
    std::int32_t rows = 0;
    if (!row_count(input_desc[0].dims, &rows)) return 1;
    const dim3 threads(kWarpSize, kWarpsPerBlock, 1);
    const int elements_per_row =
        input_desc[0].dims.d[input_desc[0].dims.nbDims - 1];
    if (elements_per_row == 128) {
      constexpr int kRowsPerWarp = 2;
      const int blocks =
          (rows + kWarpsPerBlock * kRowsPerWarp - 1) /
          (kWarpsPerBlock * kRowsPerWarp);
      exact_softmax_kernel<128, kRowsPerWarp><<<blocks, threads, 0, stream>>>(
          static_cast<const float*>(inputs[0]),
          static_cast<float*>(outputs[0]), rows);
    } else {
      const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
      exact_softmax_kernel<512, 1><<<blocks, threads, 0, stream>>>(
          static_cast<const float*>(inputs[0]),
          static_cast<float*>(outputs[0]), rows);
    }
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact Softmax launch failed: %s\n",
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

class ExactSoftmaxPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactSoftmaxPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactSoftmaxPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_softmax_register() noexcept { return 0; }
