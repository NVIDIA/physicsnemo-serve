#include "physicsnemo/inference/backend.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace physicsnemo::inference {
namespace {

std::string shape_string(const Shape& shape) {
  std::ostringstream result;
  result << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) result << ',';
    result << shape[i];
  }
  result << ']';
  return result.str();
}

}  // namespace

SessionCapabilities BackendSession::capabilities() const { return {}; }

std::vector<SharedTensor> BackendSession::run_owned(
    const std::vector<TensorView>& inputs) {
  auto produced = run(inputs);
  std::vector<SharedTensor> retained;
  retained.reserve(produced.size());
  for (auto& output : produced) {
    auto owner = std::make_shared<OwnedTensor>(std::move(output));
    const auto view = owner->view();
    std::shared_ptr<void> storage_owner = owner;
    retained.emplace_back(
        view.name, view.dtype, view.device, view.shape, view.data,
        view.byte_size, std::move(storage_owner));
  }
  return retained;
}

void BackendSession::run_into(
    const std::vector<TensorView>& inputs,
    const std::vector<MutableTensorView>& outputs) {
  const auto produced = run(inputs);
  if (produced.size() != outputs.size()) {
    throw std::runtime_error("backend output count does not match output views");
  }

  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto source = produced[i].view();
    const auto destination = outputs[i].as_read_only();
    if (source.name != destination.name) {
      throw std::runtime_error(
          "backend output name does not match caller-owned tensor: " +
          destination.name);
    }
    if (source.dtype != destination.dtype) {
      throw std::runtime_error(
          "backend output dtype does not match caller-owned tensor: " +
          destination.name);
    }
    if (source.shape != destination.shape) {
      throw std::runtime_error(
          "backend output shape does not match caller-owned tensor: " +
          destination.name + " produced=" + shape_string(source.shape) +
          " requested=" + shape_string(destination.shape));
    }
    if (source.byte_size != destination.byte_size) {
      throw std::runtime_error(
          "backend output storage does not match caller-owned tensor: " +
          destination.name);
    }
    if (source.device.type != DeviceType::kCpu ||
        destination.device.type != DeviceType::kCpu) {
      throw std::invalid_argument(
          "default caller-owned output path requires CPU tensors");
    }
    if (source.byte_size != 0) {
      std::memcpy(outputs[i].data, source.data, source.byte_size);
    }
  }
}

}  // namespace physicsnemo::inference
