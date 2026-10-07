#include "physicsnemo/inference/backends/tensorrt_exact.hpp"
#include "exact_plugin_support.hpp"

#include <NvInfer.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>

namespace physicsnemo::inference {
namespace {

constexpr char kPluginName[] = "PNMIRExactDesliceBmm";
constexpr char kPluginVersion[] = "1";

bool valid_dimensions(const nvinfer1::Dims& weights,
                      const nvinfer1::Dims& features) {
  if (weights.nbDims != 4 || features.nbDims != 4 || weights.d[0] != 1 ||
      weights.d[2] <= 1 ||
      features.d[0] != weights.d[0] || features.d[1] != weights.d[3] ||
      features.d[2] != weights.d[2]) {
    return false;
  }
  for (int32_t index = 0; index < weights.nbDims; ++index) {
    if (weights.d[index] <= 0 || features.d[index] <= 0 ||
        weights.d[index] > std::numeric_limits<int>::max() ||
        features.d[index] > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  // cuBLAS leading dimensions are also signed 32-bit integers.
  return weights.d[2] <= std::numeric_limits<int>::max() / weights.d[3] &&
         features.d[2] <= std::numeric_limits<int>::max() / features.d[3];
}

class ExactDesliceBmmPlugin final : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
 public:
  ~ExactDesliceBmmPlugin() override {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    return tensorrt_detail::capability_interface(this, type);
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) ExactDesliceBmmPlugin();
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
        nb_shape_inputs != 0 || nb_outputs != 1 || inputs[0].nbDims != 4 ||
        inputs[1].nbDims != 4) {
      return 1;
    }
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[2];
    outputs[0].d[2] = inputs[0].d[1];
    outputs[0].d[3] = inputs[1].d[3];
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
    const auto& weights = input_desc[0].dims;
    const auto& features = input_desc[1].dims;
    const int tokens = static_cast<int>(weights.d[1]);
    const int heads = static_cast<int>(weights.d[2]);
    const int slices = static_cast<int>(weights.d[3]);
    const int dimensions = static_cast<int>(features.d[3]);
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
    // Reproduce eager einsum("bths,bhsd->bthd") with weights physically BTHS
    // and attention physically BSHD. ATen folds batch/head into a strided BMM;
    // packing either operand or using eight separate GEMMs changes rounding.
    // The plugin emits packed BHTD; the graph restores logical BTHD afterwards.
    const auto status = cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, dimensions, tokens, slices, &alpha,
        static_cast<const float*>(inputs[1]), heads * dimensions, dimensions,
        static_cast<const float*>(inputs[0]), heads * slices, slices, &beta,
        static_cast<float*>(outputs[0]), dimensions,
        static_cast<long long>(tokens) * dimensions, heads);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "TensorRT exact deslice BMM cuBLAS call failed: %d\n",
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
  cublasHandle_t handle_{nullptr};
  nvinfer1::PluginFieldCollection fields_{0, nullptr};
};

class ExactDesliceBmmPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection*,
      nvinfer1::TensorRTPhase) noexcept override {
    return new (std::nothrow) ExactDesliceBmmPlugin();
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

REGISTER_TENSORRT_PLUGIN(ExactDesliceBmmPluginCreator);

}  // namespace
}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_deslice_bmm_register() noexcept { return 0; }
