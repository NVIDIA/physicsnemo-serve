#include "physicsnemo/inference/backends/onnxruntime.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace physicsnemo::inference {
namespace {

ONNXTensorElementDataType to_onnx_dtype(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::kBFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case DType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }
  throw std::invalid_argument("unsupported pnmir dtype");
}

DType from_onnx_dtype(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return DType::kBFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DType::kUInt8;
    default:
      throw std::runtime_error("ONNX Runtime returned an unsupported dtype");
  }
}

bool has_cuda_execution_provider() {
  const auto providers = Ort::GetAvailableProviders();
  return std::ranges::find(providers, "CUDAExecutionProvider") !=
         providers.end();
}

Ort::SessionOptions create_session_options(Device target) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (target.type == DeviceType::kCuda) {
    if (!has_cuda_execution_provider()) {
      throw std::runtime_error(
          "ONNX Runtime CUDA Execution Provider is unavailable");
    }
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update({
        {"device_id", std::to_string(target.index)},
        {"use_tf32", "0"},
    });
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  }
  return options;
}

Ort::MemoryInfo memory_info(Device device) {
  if (device.type == DeviceType::kCpu) {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }
  return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device.index,
                         OrtMemTypeDefault);
}

Ort::Value ort_tensor(const TensorView& tensor) {
  auto memory = memory_info(tensor.device);
  return Ort::Value::CreateTensor(
      memory, const_cast<void*>(tensor.data), tensor.byte_size,
      tensor.shape.data(), tensor.shape.size(), to_onnx_dtype(tensor.dtype));
}

Ort::Value ort_tensor(const MutableTensorView& tensor) {
  auto memory = memory_info(tensor.device);
  return Ort::Value::CreateTensor(
      memory, tensor.data, tensor.byte_size, tensor.shape.data(),
      tensor.shape.size(), to_onnx_dtype(tensor.dtype));
}

bool compatible_shape(const Shape& manifest_shape,
                      const std::vector<std::int64_t>& model_shape) {
  if (manifest_shape.size() != model_shape.size()) return false;
  for (std::size_t i = 0; i < manifest_shape.size(); ++i) {
    if (manifest_shape[i] == -1) {
      if (model_shape[i] != -1) return false;
    } else if (model_shape[i] != -1 &&
               manifest_shape[i] != model_shape[i]) {
      return false;
    }
  }
  return true;
}

void validate_contract(const Ort::Session& session,
                       const std::vector<TensorSpec>& specs, bool input) {
  const std::string kind = input ? "input" : "output";
  const std::size_t count =
      input ? session.GetInputCount() : session.GetOutputCount();
  if (count != specs.size()) {
    throw std::runtime_error("ONNX Runtime " + kind +
                             " count does not match manifest");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  std::unordered_map<std::string, std::size_t> indices;
  for (std::size_t i = 0; i < count; ++i) {
    auto allocated_name = input
                              ? session.GetInputNameAllocated(i, allocator)
                              : session.GetOutputNameAllocated(i, allocator);
    indices.emplace(allocated_name.get(), i);
  }

  for (const auto& spec : specs) {
    const auto found = indices.find(spec.name);
    if (found == indices.end()) {
      throw std::runtime_error("ONNX Runtime model has no " + kind +
                               " named: " + spec.name);
    }
    const auto type_info = input ? session.GetInputTypeInfo(found->second)
                                 : session.GetOutputTypeInfo(found->second);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
      throw std::runtime_error("ONNX Runtime " + kind +
                               " is not a tensor: " + spec.name);
    }
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != to_onnx_dtype(spec.dtype)) {
      throw std::runtime_error("ONNX Runtime " + kind +
                               " dtype mismatch for tensor: " + spec.name);
    }
    if (!compatible_shape(spec.shape, tensor_info.GetShape())) {
      throw std::runtime_error("ONNX Runtime " + kind +
                               " shape mismatch for tensor: " + spec.name);
    }
  }
}

class OnnxRuntimeSession final : public BackendSession {
 public:
  OnnxRuntimeSession(const std::filesystem::path& artifact_path,
                     const ModelManifest& manifest, Device target)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "pnmir"),
        options_(create_session_options(target)),
        session_(environment_, artifact_path.c_str(), options_),
        outputs_(manifest.outputs),
        target_(target) {
    validate_contract(session_, manifest.inputs, true);
    validate_contract(session_, manifest.outputs, false);
    input_names_.reserve(manifest.inputs.size());
    for (const auto& input : manifest.inputs) {
      input_names_.push_back(input.name);
    }
    output_names_.reserve(manifest.outputs.size());
    for (const auto& output : manifest.outputs) {
      output_names_.push_back(output.name);
    }
  }

  SessionCapabilities capabilities() const override {
    const bool cuda = target_.type == DeviceType::kCuda;
    return {
        .accepts_cpu_inputs = true,
        .accepts_device_inputs = cuda,
        .caller_owned_cpu_outputs = true,
        .caller_owned_device_outputs = cuda,
    };
  }

  std::vector<OwnedTensor> run(
      const std::vector<TensorView>& inputs) override {
    Ort::IoBinding binding(session_);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      validate_device(inputs[i].device, "input");
      ort_inputs.push_back(ort_tensor(inputs[i]));
      binding.BindInput(input_names_[i].c_str(), ort_inputs.back());
    }

    const auto cpu_memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (const auto& name : output_names_) {
      binding.BindOutput(name.c_str(), cpu_memory);
    }
    session_.Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    auto ort_outputs = binding.GetOutputValues();
    if (ort_outputs.size() != outputs_.size()) {
      throw std::runtime_error(
          "ONNX Runtime output count does not match manifest");
    }

    std::vector<OwnedTensor> outputs;
    outputs.reserve(ort_outputs.size());
    for (std::size_t i = 0; i < ort_outputs.size(); ++i) {
      if (!ort_outputs[i].IsTensor()) {
        throw std::runtime_error("ONNX Runtime returned a non-tensor output");
      }
      const auto info = ort_outputs[i].GetTensorTypeAndShapeInfo();
      Shape shape = info.GetShape();
      const DType dtype = from_onnx_dtype(info.GetElementType());
      const std::size_t bytes = info.GetElementCount() * dtype_size(dtype);
      std::vector<std::byte> storage(bytes);
      if (bytes != 0) {
        std::memcpy(storage.data(), ort_outputs[i].GetTensorRawData(), bytes);
      }
      outputs.emplace_back(outputs_[i].name, dtype, Device{}, std::move(shape),
                           std::move(storage));
    }
    return outputs;
  }

  void run_into(
      const std::vector<TensorView>& inputs,
      const std::vector<MutableTensorView>& outputs) override {
    Ort::IoBinding binding(session_);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      validate_device(inputs[i].device, "input");
      ort_inputs.push_back(ort_tensor(inputs[i]));
      binding.BindInput(input_names_[i].c_str(), ort_inputs.back());
    }

    std::vector<Ort::Value> ort_outputs;
    ort_outputs.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      validate_device(outputs[i].device, "output");
      ort_outputs.push_back(ort_tensor(outputs[i]));
      binding.BindOutput(output_names_[i].c_str(), ort_outputs.back());
    }

    session_.Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
  }

 private:
  void validate_device(Device device, const std::string& kind) const {
    if (device.type == DeviceType::kCpu) return;
    if (target_.type != DeviceType::kCuda || device != target_) {
      throw std::invalid_argument("ONNX Runtime " + kind +
                                  " device does not match the session device");
    }
  }

  Ort::Env environment_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<TensorSpec> outputs_;
  Device target_;
};

class OnnxRuntimeBackend final : public Backend {
 public:
  std::string name() const override { return "onnxruntime"; }

  bool supports(const ArtifactSpec& artifact,
                const SessionOptions& options) const override {
    const bool precision_matches =
        options.precision == "auto" || options.precision == artifact.precision;
    const bool version_matches = artifact.runtime_version.empty() ||
                                 artifact.runtime_version ==
                                     Ort::GetVersionString();
    return artifact.backend == name() &&
           artifact.target == options.device.type && precision_matches &&
           version_matches;
  }

  std::unique_ptr<BackendSession> create_session(
      const ModelPackage& package, const ArtifactSpec& artifact,
      const SessionOptions& options) const override {
    return std::make_unique<OnnxRuntimeSession>(
        package.artifact_path(artifact), package.manifest(), options.device);
  }
};

}  // namespace

std::unique_ptr<Backend> create_onnxruntime_backend() {
  return std::make_unique<OnnxRuntimeBackend>();
}

}  // namespace physicsnemo::inference
