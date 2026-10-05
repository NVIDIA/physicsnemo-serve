#include "physicsnemo/inference/backends/aoti.hpp"
#include "physicsnemo/inference/runtime.hpp"

#ifdef PNMIR_HAS_DOMINO_EXACT_OPS
#include "physicsnemo/inference/backends/domino_exact_ops.hpp"
#endif

#include <ATen/Context.h>
#ifdef PNMIR_TORCH_HAS_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace physicsnemo::inference {
namespace {

at::ScalarType to_torch_dtype(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return at::kFloat;
    case DType::kFloat16:
      return at::kHalf;
    case DType::kBFloat16:
      return at::kBFloat16;
    case DType::kInt32:
      return at::kInt;
    case DType::kInt64:
      return at::kLong;
    case DType::kUInt8:
      return at::kByte;
  }
  throw std::invalid_argument("unsupported pnmir dtype");
}

DType from_torch_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return DType::kFloat32;
    case at::kHalf:
      return DType::kFloat16;
    case at::kBFloat16:
      return DType::kBFloat16;
    case at::kInt:
      return DType::kInt32;
    case at::kLong:
      return DType::kInt64;
    case at::kByte:
      return DType::kUInt8;
    default:
      throw std::runtime_error("AOTInductor returned an unsupported dtype");
  }
}

std::vector<std::int64_t> torch_shape(const TensorView& tensor) {
  return tensor.shape;
}

c10::DeviceIndex loader_device_index(Device target) {
  if (target.type != DeviceType::kCuda) return -1;
  if (target.index < 0 ||
      target.index > std::numeric_limits<c10::DeviceIndex>::max()) {
    throw std::invalid_argument("AOTInductor CUDA device index is out of range");
  }
  return static_cast<c10::DeviceIndex>(target.index);
}

std::string prepare_artifact(const std::filesystem::path& artifact_path,
                             Device target) {
  if (target.type == DeviceType::kCuda) {
    // Match the exporter for both matmuls and ATen fallback convolutions.
    // cuDNN has an independent TF32 setting consulted by DoMINO's convolutions.
    at::globalContext().setFloat32MatmulPrecision("highest");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
  }
  return artifact_path.string();
}

class AOTISession final : public BackendSession {
 public:
  AOTISession(std::filesystem::path artifact_path,
              std::vector<TensorSpec> output_specs, Device target)
      : loader_(prepare_artifact(artifact_path, target), "model", false, 1,
                loader_device_index(target)),
        outputs_(std::move(output_specs)),
        target_(target) {}

  SessionCapabilities capabilities() const override {
    return {
        .accepts_cpu_inputs = true,
        .accepts_device_inputs = target_.type == DeviceType::kCuda,
        .caller_owned_cpu_outputs = true,
        .caller_owned_device_outputs = false,
        .backend_owned_device_outputs = target_.type == DeviceType::kCuda,
    };
  }

  std::vector<OwnedTensor> run(
      const std::vector<TensorView>& inputs) override {
    auto torch_outputs = execute(inputs);
    std::vector<OwnedTensor> outputs;
    outputs.reserve(torch_outputs.size());
    for (std::size_t i = 0; i < torch_outputs.size(); ++i) {
      auto tensor = torch_outputs[i].detach().to(torch::kCPU).contiguous();
      Shape shape(tensor.sizes().begin(), tensor.sizes().end());
      const DType dtype = from_torch_dtype(tensor.scalar_type());
      const std::size_t bytes =
          static_cast<std::size_t>(tensor.numel()) * dtype_size(dtype);
      std::vector<std::byte> storage(bytes);
      if (bytes != 0) {
        std::memcpy(storage.data(), tensor.const_data_ptr(), bytes);
      }
      outputs.emplace_back(outputs_[i].name, dtype, Device{}, std::move(shape),
                           std::move(storage));
    }
    return outputs;
  }

  std::vector<SharedTensor> run_owned(
      const std::vector<TensorView>& inputs) override {
    auto torch_outputs = execute(inputs);
    std::vector<SharedTensor> outputs;
    outputs.reserve(torch_outputs.size());
    for (std::size_t i = 0; i < torch_outputs.size(); ++i) {
      auto tensor = torch_outputs[i].detach().contiguous();
      Shape shape(tensor.sizes().begin(), tensor.sizes().end());
      const DType dtype = from_torch_dtype(tensor.scalar_type());
      const std::size_t bytes =
          static_cast<std::size_t>(tensor.numel()) * dtype_size(dtype);
      const Device device =
          tensor.is_cuda()
              ? Device{DeviceType::kCuda, tensor.get_device()}
              : Device{};
      // detach()/contiguous() can retain from_blob's non-owning input storage.
      // Copy those aliases, including offset views, before the caller may reuse
      // its allocation. Other outputs already retain their Torch-owned storage.
      const auto output_begin =
          reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr());
      for (const auto& input : inputs) {
        if (bytes == 0 || input.byte_size == 0 ||
            input.device.type != device.type ||
            (device.type == DeviceType::kCuda &&
             input.device.index != device.index)) {
          continue;
        }
        const auto input_begin =
            reinterpret_cast<std::uintptr_t>(input.data);
        const bool overlaps = output_begin >= input_begin
                                  ? output_begin - input_begin < input.byte_size
                                  : input_begin - output_begin < bytes;
        if (overlaps) {
          tensor = tensor.clone();
          break;
        }
      }
      auto owner = std::make_shared<at::Tensor>(std::move(tensor));
      const void* data = owner->const_data_ptr();
      std::shared_ptr<void> storage_owner = owner;
      outputs.emplace_back(outputs_[i].name, dtype, device, std::move(shape),
                           data, bytes, std::move(storage_owner));
    }
#ifdef PNMIR_TORCH_HAS_CUDA
    if (target_.type == DeviceType::kCuda) {
      const c10::cuda::CUDAGuard guard(target_.index);
      c10::cuda::device_synchronize();
    }
#endif
    return outputs;
  }

 private:
  std::vector<at::Tensor> execute(
      const std::vector<TensorView>& inputs) {
#ifdef PNMIR_TORCH_HAS_CUDA
    // AOTI selects its launch stream from the current CUDA device.
    c10::cuda::OptionalCUDAGuard guard;
    if (target_.type == DeviceType::kCuda) {
      guard.set_index(static_cast<c10::DeviceIndex>(target_.index));
    }
#endif
    c10::InferenceMode inference_mode;
    std::vector<at::Tensor> torch_inputs;
    torch_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.device.type == DeviceType::kCuda) {
        if (target_.type != DeviceType::kCuda ||
            input.device.index != target_.index) {
          throw std::invalid_argument(
              "AOTInductor input device does not match the session device");
        }
        const auto options =
            torch::TensorOptions()
                .dtype(to_torch_dtype(input.dtype))
                .device(torch::Device(torch::kCUDA, target_.index));
        torch_inputs.push_back(torch::from_blob(
            const_cast<void*>(input.data), torch_shape(input), options));
        continue;
      }
      if (input.device.type != DeviceType::kCpu) {
        throw std::invalid_argument(
            "AOTInductor input device is unsupported");
      }
      const auto options =
          torch::TensorOptions().dtype(to_torch_dtype(input.dtype)).device(torch::kCPU);
      auto tensor = torch::from_blob(const_cast<void*>(input.data),
                                     torch_shape(input), options);
      if (target_.type == DeviceType::kCuda) {
        tensor = tensor.to(torch::Device(torch::kCUDA, target_.index));
      }
      torch_inputs.push_back(std::move(tensor));
    }

    auto torch_outputs = loader_.run(torch_inputs);
    if (torch_outputs.size() != outputs_.size()) {
      throw std::runtime_error("AOTInductor output count does not match manifest");
    }
    return torch_outputs;
  }

  torch::inductor::AOTIModelPackageLoader loader_;
  std::vector<TensorSpec> outputs_;
  Device target_;
};

class AOTIBackend final : public Backend {
 public:
  std::string name() const override { return "aoti"; }

  bool supports(const ArtifactSpec& artifact,
                const SessionOptions& options) const override {
    const bool precision_matches =
        options.precision == "auto" || options.precision == artifact.precision;
    const bool version_matches = artifact.runtime_version.empty() ||
                                 artifact.runtime_version ==
                                     PNMIR_TORCH_RUNTIME_VERSION;
    return artifact.backend == name() &&
           artifact.target == options.device.type && precision_matches &&
           version_matches;
  }

  std::unique_ptr<BackendSession> create_session(
      const ModelPackage& package, const ArtifactSpec& artifact,
      const SessionOptions& options) const override {
#ifdef PNMIR_TORCH_HAS_CUDA
    // AOTI's generated constructor changes the current CUDA device while
    // loading constants; restore the caller's device even if loading fails.
    c10::cuda::OptionalCUDAGuard guard;
    if (options.device.type == DeviceType::kCuda) {
      guard.set_index(loader_device_index(options.device));
    }
#endif
    return std::make_unique<AOTISession>(package.artifact_path(artifact),
                                         package.manifest().outputs,
                                         options.device);
  }
};

}  // namespace

std::unique_ptr<Backend> create_aoti_backend() {
  return std::make_unique<AOTIBackend>();
}

void register_aoti_exact_operators(Runtime& runtime) {
#ifdef PNMIR_HAS_DOMINO_EXACT_OPS
  domino::register_exact_ops();
  runtime.register_operator(std::string(domino::kExactOperatorId),
                            std::string(domino::kExactOperatorAbi));
#else
  static_cast<void>(runtime);
#endif
}

}  // namespace physicsnemo::inference
