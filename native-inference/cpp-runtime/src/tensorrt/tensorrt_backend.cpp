#include "physicsnemo/inference/backends/tensorrt.hpp"
#include "physicsnemo/inference/runtime.hpp"

#ifdef PNMIR_HAS_TENSORRT_EXACT
#include "physicsnemo/inference/backends/tensorrt_exact_attention_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_deslice_bmm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_gelu_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_gemm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_inverse_distance_blend_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_layer_norm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_linear_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_scalar_div_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_slice_bmm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_softmax_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_token_sum_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_weighted_blend_plugin.hpp"
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace physicsnemo::inference {
namespace {

void register_exact_plugins() {
#ifdef PNMIR_HAS_TENSORRT_EXACT
  // References keep each plugin DSO loaded so its TensorRT creator is present
  // before deserializing a plan. Registration entry points are ABI-stable.
  if (pnmir_tensorrt_exact_linear_register() != 0 ||
      pnmir_tensorrt_exact_gemm_register() != 0 ||
      pnmir_tensorrt_exact_token_sum_register() != 0 ||
      pnmir_tensorrt_exact_slice_bmm_register() != 0 ||
      pnmir_tensorrt_exact_deslice_bmm_register() != 0 ||
      pnmir_tensorrt_exact_layer_norm_register() != 0 ||
      pnmir_tensorrt_exact_softmax_register() != 0 ||
      pnmir_tensorrt_exact_attention_register() != 0 ||
      pnmir_tensorrt_exact_gelu_register() != 0 ||
      pnmir_tensorrt_exact_weighted_blend_register() != 0 ||
      pnmir_tensorrt_exact_scalar_div_register() != 0 ||
      pnmir_tensorrt_exact_inverse_distance_blend_register() != 0) {
    throw std::runtime_error("TensorRT exact plugin registration failed");
  }
#endif
}

void check_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " +
                             cudaGetErrorString(status));
  }
}

std::string tensorrt_runtime_version() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH) + "." +
         std::to_string(NV_TENSORRT_BUILD);
}

nvinfer1::DataType to_tensorrt_dtype(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return nvinfer1::DataType::kFLOAT;
    case DType::kFloat16:
      return nvinfer1::DataType::kHALF;
    case DType::kBFloat16:
      return nvinfer1::DataType::kBF16;
    case DType::kInt32:
      return nvinfer1::DataType::kINT32;
    case DType::kInt64:
      return nvinfer1::DataType::kINT64;
    case DType::kUInt8:
      return nvinfer1::DataType::kUINT8;
  }
  throw std::invalid_argument("unsupported pnmir dtype");
}

class TensorRTLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR && message != nullptr) {
      try {
        last_error_ = message;
      } catch (...) {
      }
    }
  }

  const std::string& last_error() const { return last_error_; }

 private:
  std::string last_error_;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() {
    if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  void create() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    check_cuda(cudaMalloc(&data_, size_), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) static_cast<void>(cudaFree(data_));
  }

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  void* data_{nullptr};
  std::size_t size_{0};
};

std::vector<std::byte> read_engine(
    const std::filesystem::path& artifact_path) {
  std::ifstream input(artifact_path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open TensorRT engine: " +
                             artifact_path.string());
  }
  const auto end = input.tellg();
  if (end <= 0) {
    throw std::runtime_error("TensorRT engine is empty: " +
                             artifact_path.string());
  }
  std::vector<std::byte> data(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(data.size()))) {
    throw std::runtime_error("cannot read TensorRT engine: " +
                             artifact_path.string());
  }
  return data;
}

bool compatible_shape(const Shape& manifest_shape,
                      const nvinfer1::Dims64& engine_shape) {
  if (engine_shape.nbDims < 0 ||
      manifest_shape.size() !=
          static_cast<std::size_t>(engine_shape.nbDims)) {
    return false;
  }
  for (std::size_t i = 0; i < manifest_shape.size(); ++i) {
    if (manifest_shape[i] <= 0 ||
        engine_shape.d[i] != manifest_shape[i]) {
      return false;
    }
  }
  return true;
}

void validate_contract(const nvinfer1::ICudaEngine& engine,
                       const std::vector<TensorSpec>& specs, bool input) {
  const std::string kind = input ? "input" : "output";
  const auto expected_mode = input ? nvinfer1::TensorIOMode::kINPUT
                                   : nvinfer1::TensorIOMode::kOUTPUT;
  std::unordered_map<std::string, nvinfer1::TensorIOMode> tensors;
  int count = 0;
  for (int index = 0; index < engine.getNbIOTensors(); ++index) {
    const char* name = engine.getIOTensorName(index);
    if (name == nullptr) {
      throw std::runtime_error("TensorRT returned an unnamed I/O tensor");
    }
    const auto mode = engine.getTensorIOMode(name);
    tensors.emplace(name, mode);
    if (mode == expected_mode) ++count;
  }
  if (count != static_cast<int>(specs.size())) {
    throw std::runtime_error("TensorRT " + kind +
                             " count does not match manifest");
  }

  for (const auto& spec : specs) {
    const auto found = tensors.find(spec.name);
    if (found == tensors.end() || found->second != expected_mode) {
      throw std::runtime_error("TensorRT engine has no " + kind +
                               " named: " + spec.name);
    }
    if (engine.getTensorDataType(spec.name.c_str()) !=
        to_tensorrt_dtype(spec.dtype)) {
      throw std::runtime_error("TensorRT " + kind +
                               " dtype mismatch for tensor: " + spec.name);
    }
    if (!compatible_shape(spec.shape,
                          engine.getTensorShape(spec.name.c_str()))) {
      throw std::runtime_error("TensorRT " + kind +
                               " shape mismatch for tensor: " + spec.name);
    }
    if (engine.getTensorLocation(spec.name.c_str()) !=
        nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error("TensorRT " + kind +
                               " must use device memory: " + spec.name);
    }
  }
}

std::size_t tensor_bytes(const TensorSpec& spec) {
  return element_count(spec.shape) * dtype_size(spec.dtype);
}

class TensorRTSession final : public BackendSession {
 public:
  TensorRTSession(const std::filesystem::path& artifact_path,
                  const ModelManifest& manifest, Device target)
      : inputs_(manifest.inputs),
        outputs_(manifest.outputs),
        target_(target) {
    check_cuda(cudaSetDevice(target_.index), "cudaSetDevice");
    stream_.create();
    if (!initLibNvInferPlugins(&logger_, "")) {
      throw std::runtime_error("TensorRT plugin registration failed");
    }
    register_exact_plugins();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("TensorRT runtime creation failed");
    }
    const auto engine_data = read_engine(artifact_path);
    engine_.reset(
        runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
      const std::string detail =
          logger_.last_error().empty() ? "" : ": " + logger_.last_error();
      throw std::runtime_error("TensorRT could not deserialize engine" + detail);
    }

    validate_contract(*engine_, inputs_, true);
    validate_contract(*engine_, outputs_, false);
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("TensorRT execution context creation failed");
    }

    input_buffers_.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      input_buffers_.emplace_back(tensor_bytes(input));
      if (!context_->setTensorAddress(input.name.c_str(),
                                      input_buffers_.back().data())) {
        throw std::runtime_error("TensorRT could not bind input: " +
                                 input.name);
      }
    }
    output_buffers_.reserve(outputs_.size());
    for (const auto& output : outputs_) {
      output_buffers_.emplace_back(tensor_bytes(output));
      if (!context_->setTensorAddress(output.name.c_str(),
                                      output_buffers_.back().data())) {
        throw std::runtime_error("TensorRT could not bind output: " +
                                 output.name);
      }
    }
  }

  SessionCapabilities capabilities() const override {
    return {
        .accepts_cpu_inputs = true,
        .accepts_device_inputs = true,
        .caller_owned_cpu_outputs = true,
        .caller_owned_device_outputs = true,
    };
  }

  std::vector<OwnedTensor> run(
      const std::vector<TensorView>& inputs) override {
    std::vector<std::vector<std::byte>> host_outputs;
    host_outputs.reserve(outputs_.size());
    std::vector<MutableTensorView> output_views;
    output_views.reserve(outputs_.size());
    for (const auto& output : outputs_) {
      host_outputs.emplace_back(tensor_bytes(output));
      output_views.push_back(
          {output.name, output.dtype, Device{}, output.shape,
           host_outputs.back().data(), host_outputs.back().size()});
    }
    run_into(inputs, output_views);

    std::vector<OwnedTensor> result;
    result.reserve(outputs_.size());
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      result.emplace_back(outputs_[i].name, outputs_[i].dtype, Device{},
                          outputs_[i].shape, std::move(host_outputs[i]));
    }
    return result;
  }

  void run_into(const std::vector<TensorView>& inputs,
                const std::vector<MutableTensorView>& outputs) override {
    check_cuda(cudaSetDevice(target_.index), "cudaSetDevice");
    if (inputs.size() != input_buffers_.size()) {
      throw std::invalid_argument(
          "TensorRT input count does not match manifest");
    }
    if (outputs.size() != output_buffers_.size()) {
      throw std::invalid_argument(
          "TensorRT output count does not match manifest");
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      void* address = input_buffers_[i].data();
      if (inputs[i].device.type == DeviceType::kCpu) {
        check_cuda(cudaMemcpyAsync(address, inputs[i].data,
                                   input_buffers_[i].size(),
                                   cudaMemcpyHostToDevice, stream_.get()),
                   "TensorRT input copy");
      } else if (inputs[i].device.type == DeviceType::kCuda &&
                 inputs[i].device.index == target_.index) {
        address = const_cast<void*>(inputs[i].data);
      } else {
        throw std::invalid_argument(
            "TensorRT input device does not match the session device");
      }
      if (!context_->setTensorAddress(inputs_[i].name.c_str(), address)) {
        throw std::runtime_error("TensorRT could not bind input: " +
                                 inputs_[i].name);
      }
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
      void* address = output_buffers_[i].data();
      if (outputs[i].device.type == DeviceType::kCuda) {
        if (outputs[i].device.index != target_.index) {
          throw std::invalid_argument(
              "TensorRT output device does not match the session device");
        }
        address = outputs[i].data;
      } else if (outputs[i].device.type != DeviceType::kCpu) {
        throw std::invalid_argument("TensorRT output device is unsupported");
      }
      if (!context_->setTensorAddress(outputs_[i].name.c_str(), address)) {
        throw std::runtime_error("TensorRT could not bind output: " +
                                 outputs_[i].name);
      }
    }

    if (!context_->enqueueV3(stream_.get())) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs[i].device.type == DeviceType::kCpu) {
        check_cuda(cudaMemcpyAsync(outputs[i].data, output_buffers_[i].data(),
                                   output_buffers_[i].size(),
                                   cudaMemcpyDeviceToHost, stream_.get()),
                   "TensorRT output copy");
      }
    }
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "TensorRT stream synchronization");
  }

 private:
  TensorRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  CudaStream stream_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  Device target_;
  std::vector<DeviceBuffer> input_buffers_;
  std::vector<DeviceBuffer> output_buffers_;
};

class TensorRTBackend final : public Backend {
 public:
  std::string name() const override { return "tensorrt"; }

  bool supports(const ArtifactSpec& artifact,
                const SessionOptions& options) const override {
    const bool precision_matches =
        options.precision == "auto" || options.precision == artifact.precision;
    const bool version_matches =
        artifact.runtime_version.empty() ||
        artifact.runtime_version == tensorrt_runtime_version();
    return artifact.backend == name() &&
           artifact.target == DeviceType::kCuda &&
           options.device.type == DeviceType::kCuda && precision_matches &&
           version_matches;
  }

  std::unique_ptr<BackendSession> create_session(
      const ModelPackage& package, const ArtifactSpec& artifact,
      const SessionOptions& options) const override {
    return std::make_unique<TensorRTSession>(
        package.artifact_path(artifact), package.manifest(), options.device);
  }
};

}  // namespace

std::unique_ptr<Backend> create_tensorrt_backend() {
  return std::make_unique<TensorRTBackend>();
}

void register_tensorrt_exact_operators(Runtime& runtime) {
#ifdef PNMIR_HAS_TENSORRT_EXACT
  register_exact_plugins();
  runtime.register_operator(std::string(kTensorRTExactLinearOperatorId),
                            std::string(kTensorRTExactLinearOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactGemmOperatorId),
                            std::string(kTensorRTExactGemmOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactTokenSumOperatorId),
                            std::string(kTensorRTExactTokenSumOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactSliceBmmOperatorId),
                            std::string(kTensorRTExactSliceBmmOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactDesliceBmmOperatorId),
                            std::string(kTensorRTExactDesliceBmmOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactLayerNormOperatorId),
                            std::string(kTensorRTExactLayerNormOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactSoftmaxOperatorId),
                            std::string(kTensorRTExactSoftmaxOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactAttentionOperatorId),
                            std::string(kTensorRTExactAttentionOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactGeluOperatorId),
                            std::string(kTensorRTExactGeluOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactWeightedBlendOperatorId),
                            std::string(kTensorRTExactWeightedBlendOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactScalarDivOperatorId),
                            std::string(kTensorRTExactScalarDivOperatorAbi));
  runtime.register_operator(std::string(kTensorRTExactInverseDistanceBlendOperatorId),
                            std::string(kTensorRTExactInverseDistanceBlendOperatorAbi));
#else
  static_cast<void>(runtime);
#endif
}

}  // namespace physicsnemo::inference
