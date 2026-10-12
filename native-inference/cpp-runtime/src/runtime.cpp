#include "physicsnemo/inference/runtime.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace physicsnemo::inference {
namespace {

void validate_tensor(const TensorView& tensor, const TensorSpec& spec) {
  if (tensor.name != spec.name) {
    throw std::invalid_argument("expected tensor '" + spec.name +
                                "', got '" + tensor.name + "'");
  }
  if (tensor.dtype != spec.dtype) {
    throw std::invalid_argument("dtype mismatch for tensor: " + spec.name);
  }
  if (tensor.shape.size() != spec.shape.size()) {
    throw std::invalid_argument("rank mismatch for tensor: " + spec.name);
  }
  for (std::size_t i = 0; i < spec.shape.size(); ++i) {
    if (tensor.shape[i] <= 0 ||
        (spec.shape[i] != -1 && tensor.shape[i] != spec.shape[i])) {
      throw std::invalid_argument("shape mismatch for tensor: " + spec.name);
    }
  }
  const std::size_t expected = tensor_byte_size(tensor.shape, tensor.dtype);
  if (tensor.byte_size != expected || (expected != 0 && tensor.data == nullptr)) {
    throw std::invalid_argument("storage mismatch for tensor: " + spec.name);
  }
}

std::vector<TensorView> order_inputs(const std::vector<TensorView>& inputs,
                                     const std::vector<TensorSpec>& specs) {
  if (inputs.size() != specs.size()) {
    throw std::invalid_argument("input tensor count does not match manifest");
  }

  std::unordered_map<std::string, const TensorView*> by_name;
  for (const auto& input : inputs) {
    if (!by_name.emplace(input.name, &input).second) {
      throw std::invalid_argument("duplicate input tensor: " + input.name);
    }
  }

  std::vector<TensorView> ordered;
  ordered.reserve(specs.size());
  for (const auto& spec : specs) {
    const auto found = by_name.find(spec.name);
    if (found == by_name.end()) {
      throw std::invalid_argument("missing input tensor: " + spec.name);
    }
    validate_tensor(*found->second, spec);
    ordered.push_back(*found->second);
  }
  return ordered;
}

std::vector<MutableTensorView> order_outputs(
    const std::vector<MutableTensorView>& outputs,
    const std::vector<TensorSpec>& specs) {
  if (outputs.size() != specs.size()) {
    throw std::invalid_argument("output tensor count does not match manifest");
  }

  std::unordered_map<std::string, const MutableTensorView*> by_name;
  for (const auto& output : outputs) {
    if (!by_name.emplace(output.name, &output).second) {
      throw std::invalid_argument("duplicate output tensor: " + output.name);
    }
  }

  std::vector<MutableTensorView> ordered;
  ordered.reserve(specs.size());
  for (const auto& spec : specs) {
    const auto found = by_name.find(spec.name);
    if (found == by_name.end()) {
      throw std::invalid_argument("missing output tensor: " + spec.name);
    }
    validate_tensor(found->second->as_read_only(), spec);
    ordered.push_back(*found->second);
  }
  return ordered;
}

}  // namespace

InferenceSession::InferenceSession(
    ModelManifest manifest, std::unique_ptr<BackendSession> backend_session,
    std::string backend_name)
    : manifest_(std::move(manifest)),
      backend_session_(std::move(backend_session)),
      backend_name_(std::move(backend_name)) {}

SessionCapabilities InferenceSession::capabilities() const {
  return backend_session_->capabilities();
}

std::vector<OwnedTensor> InferenceSession::run(
    const std::vector<TensorView>& inputs) {
  const auto ordered = order_inputs(inputs, manifest_.inputs);
  auto outputs = backend_session_->run(ordered);
  if (outputs.size() != manifest_.outputs.size()) {
    throw std::runtime_error("backend output count does not match manifest");
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    validate_tensor(outputs[i].view(), manifest_.outputs[i]);
  }
  return outputs;
}

std::vector<SharedTensor> InferenceSession::run_owned(
    const std::vector<TensorView>& inputs) {
  const auto ordered = order_inputs(inputs, manifest_.inputs);
  auto outputs = backend_session_->run_owned(ordered);
  if (outputs.size() != manifest_.outputs.size()) {
    throw std::runtime_error("backend output count does not match manifest");
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    validate_tensor(outputs[i].view(), manifest_.outputs[i]);
  }
  return outputs;
}

void InferenceSession::run_into(
    const std::vector<TensorView>& inputs,
    const std::vector<MutableTensorView>& outputs) {
  const auto ordered_inputs = order_inputs(inputs, manifest_.inputs);
  const auto ordered_outputs = order_outputs(outputs, manifest_.outputs);
  const auto supported = backend_session_->capabilities();
  for (const auto& input : ordered_inputs) {
    if (input.device.type == DeviceType::kCpu &&
        !supported.accepts_cpu_inputs) {
      throw std::invalid_argument(
          "backend does not accept caller-owned CPU inputs");
    }
    if (input.device.type != DeviceType::kCpu &&
        !supported.accepts_device_inputs) {
      throw std::invalid_argument(
          "backend does not accept caller-owned device inputs");
    }
  }
  for (const auto& output : ordered_outputs) {
    if (output.device.type == DeviceType::kCpu &&
        !supported.caller_owned_cpu_outputs) {
      throw std::invalid_argument(
          "backend does not accept caller-owned CPU outputs");
    }
    if (output.device.type != DeviceType::kCpu &&
        !supported.caller_owned_device_outputs) {
      throw std::invalid_argument(
          "backend does not accept caller-owned device outputs");
    }
  }
  backend_session_->run_into(ordered_inputs, ordered_outputs);
}

void Runtime::register_backend(std::unique_ptr<Backend> backend) {
  if (!backend) throw std::invalid_argument("backend cannot be null");
  const std::string backend_name = backend->name();
  const auto duplicate = std::ranges::find_if(
      backends_, [&](const auto& existing) {
        return existing->name() == backend_name;
      });
  if (duplicate != backends_.end()) {
    throw std::invalid_argument("backend already registered: " + backend_name);
  }
  backends_.push_back(std::move(backend));
}

void Runtime::register_operator(std::string id, std::string abi) {
  if (id.empty() || abi.empty()) {
    throw std::invalid_argument("operator id and ABI cannot be empty");
  }
  if (!operators_.emplace(id, abi).second) {
    throw std::invalid_argument("operator already registered: " + id);
  }
}

std::unique_ptr<InferenceSession> Runtime::create_session(
    const ModelPackage& package, const SessionOptions& options) const {
  std::string unavailable_operator;
  for (const auto& artifact : package.manifest().artifacts) {
    if (options.backend && artifact.backend != *options.backend) continue;
    for (const auto& backend : backends_) {
      if (backend->name() != artifact.backend) continue;
      if (!backend->supports(artifact, options)) continue;
      const auto unavailable = std::ranges::find_if(
          artifact.required_operators, [&](const auto& required) {
            const auto found = operators_.find(required.id);
            return found == operators_.end() || found->second != required.abi;
          });
      if (unavailable != artifact.required_operators.end()) {
        unavailable_operator = unavailable->id + "@" + unavailable->abi;
        continue;
      }
      auto backend_session =
          backend->create_session(package, artifact, options);
      return std::make_unique<InferenceSession>(
          package.manifest(), std::move(backend_session), backend->name());
    }
  }

  if (!unavailable_operator.empty()) {
    throw std::runtime_error("required operator is not registered: " +
                             unavailable_operator);
  }
  const std::string requested = options.backend.value_or("auto");
  throw std::runtime_error("no compatible backend artifact for: " + requested);
}

}  // namespace physicsnemo::inference
