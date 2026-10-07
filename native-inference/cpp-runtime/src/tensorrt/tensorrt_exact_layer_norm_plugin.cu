#include "physicsnemo/inference/backends/tensorrt_exact.hpp"
#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactLayerNorm";
constexpr char kPluginVersion[] = "1";
constexpr char kEpsilonField[] = "epsilon";
constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kVectorSize = 4;

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T values[Size];
};

struct WelfordData {
  float mean;
  float sigma2;
  float count;
};

__device__ WelfordData welford_online_sum(float value,
                                           const WelfordData& current) {
  const float delta = value - current.mean;
  const float new_count = current.count + 1.0F;
  const float new_mean = current.mean + delta * (1.0F / new_count);
  return {new_mean,
          current.sigma2 + delta * (value - new_mean), new_count};
}

__device__ WelfordData welford_combine(const WelfordData data_b,
                                        const WelfordData data_a) {
  const float delta = data_b.mean - data_a.mean;
  const float count = data_a.count + data_b.count;
  if (count <= 0.0F) return {0.0F, 0.0F, 0.0F};
  const float coefficient = 1.0F / count;
  const float count_a = data_a.count * coefficient;
  const float count_b = data_b.count * coefficient;
  const float mean = count_a * data_a.mean + count_b * data_b.mean;
  const float sigma2 = data_a.sigma2 + data_b.sigma2 +
                       delta * delta * data_a.count * count_b;
  return {mean, sigma2, count};
}

__device__ WelfordData compute_stats(const float* input, int size,
                                      float* shared) {
  using Vector = AlignedVector<float, kVectorSize>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(input);
  const int threads = blockDim.x * blockDim.y;
  const int thread = threadIdx.x + threadIdx.y * blockDim.x;
  const int vector_count = size / kVectorSize;
  WelfordData value{0.0F, 0.0F, 0.0F};
  for (int index = thread; index < vector_count; index += threads) {
    const Vector data = input_vectors[index];
#pragma unroll
    for (int element = 0; element < kVectorSize; ++element) {
      value = welford_online_sum(data.values[element], value);
    }
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const WelfordData other{
        __shfl_down_sync(0xFFFFFFFFU, value.mean, offset),
        __shfl_down_sync(0xFFFFFFFFU, value.sigma2, offset),
        __shfl_down_sync(0xFFFFFFFFU, value.count, offset)};
    value = welford_combine(value, other);
  }

  if (blockDim.y > 1) {
    float* mean_sigma = shared;
    float* counts = shared + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const int destination = threadIdx.y - offset;
        mean_sigma[2 * destination] = value.mean;
        mean_sigma[2 * destination + 1] = value.sigma2;
        counts[destination] = value.count;
      }
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        const WelfordData other{mean_sigma[2 * threadIdx.y],
                                mean_sigma[2 * threadIdx.y + 1],
                                counts[threadIdx.y]};
        value = welford_combine(value, other);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean_sigma[0] = value.mean;
      mean_sigma[1] = value.sigma2 / static_cast<float>(size);
    }
    __syncthreads();
    return {mean_sigma[0], mean_sigma[1], 0.0F};
  }

  return {__shfl_sync(0xFFFFFFFFU, value.mean, 0),
          __shfl_sync(0xFFFFFFFFU, value.sigma2, 0) /
              static_cast<float>(size),
          0.0F};
}

__global__ void exact_layer_norm_kernel(int size, float epsilon,
                                        const float* input,
                                        const float* gamma,
                                        const float* beta, float* output) {
  extern __shared__ float shared[];
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  const float* input_row = input + row * size;
  const WelfordData stats = compute_stats(input_row, size, shared);
  const float reciprocal_std = rsqrtf(stats.sigma2 + epsilon);

  using Vector = AlignedVector<float, kVectorSize>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(input_row);
  const auto* gamma_vectors = reinterpret_cast<const Vector*>(gamma);
  const auto* beta_vectors = reinterpret_cast<const Vector*>(beta);
  auto* output_vectors = reinterpret_cast<Vector*>(output + row * size);
  const int threads = blockDim.x * blockDim.y;
  const int thread = threadIdx.x + threadIdx.y * blockDim.x;
  const int vector_count = size / kVectorSize;
  for (int index = thread; index < vector_count; index += threads) {
    const Vector data = input_vectors[index];
    Vector result{};
#pragma unroll
    for (int element = 0; element < kVectorSize; ++element) {
      result.values[element] =
          gamma_vectors[index].values[element] *
              (reciprocal_std * (data.values[element] - stats.mean)) +
          beta_vectors[index].values[element];
    }
    output_vectors[index] = result;
  }
}

bool valid_dimensions(const nvinfer1::Dims& input,
                      const nvinfer1::Dims& gamma,
                      const nvinfer1::Dims& beta) {
  if (input.nbDims < 2 || gamma.nbDims != 1 || beta.nbDims != 1) return false;
  const auto size = input.d[input.nbDims - 1];
  return size > 0 && size <= std::numeric_limits<int>::max() &&
         size % kVectorSize == 0 && gamma.d[0] == size &&
         beta.d[0] == size;
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

float parse_epsilon(const nvinfer1::PluginFieldCollection& fields) {
  for (int32_t index = 0; index < fields.nbFields; ++index) {
    const auto& field = fields.fields[index];
    if (field.name != nullptr && std::string_view(field.name) == kEpsilonField &&
        field.type == nvinfer1::PluginFieldType::kFLOAT32 &&
        field.length == 1 && field.data != nullptr) {
      return *static_cast<const float*>(field.data);
    }
  }
  return 1.0e-5F;
}

class ExactLayerNormPlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit ExactLayerNormPlugin(float epsilon) : epsilon_(epsilon) {
    serialization_fields_[0] = {kEpsilonField, &epsilon_,
                                nvinfer1::PluginFieldType::kFLOAT32, 1};
    serialization_collection_.nbFields = 1;
    serialization_collection_.fields = serialization_fields_.data();
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactLayerNormPlugin(epsilon_);
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
    if (!row_count(input_desc[0].dims, &rows)) return 1;
    const int size = static_cast<int>(
        input_desc[0].dims.d[input_desc[0].dims.nbDims - 1]);
    const dim3 threads(kWarpSize, kThreads / kWarpSize, 1);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads.y) * 3U / 2U * sizeof(float);
    exact_layer_norm_kernel<<<rows, threads, shared_bytes, stream>>>(
        size, epsilon_, static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<float*>(outputs[0]));
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      std::fprintf(stderr, "TensorRT exact LayerNorm launch failed: %s\n",
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
    return &serialization_collection_;
  }

 private:
  float epsilon_;
  std::array<nvinfer1::PluginField, 1> serialization_fields_{};
  nvinfer1::PluginFieldCollection serialization_collection_{};
};

class ExactLayerNormPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  ExactLayerNormPluginCreator() {
    fields_[0] = {kEpsilonField, nullptr,
                  nvinfer1::PluginFieldType::kFLOAT32, 1};
    collection_.nbFields = 1;
    collection_.fields = fields_.data();
  }

  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase) noexcept override {
    if (fields == nullptr) return nullptr;
    return new (std::nothrow) ExactLayerNormPlugin(parse_epsilon(*fields));
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
  std::array<nvinfer1::PluginField, 1> fields_{};
  nvinfer1::PluginFieldCollection collection_{};
};

REGISTER_TENSORRT_PLUGIN(ExactLayerNormPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_layer_norm_register() noexcept {
  return 0;
}
