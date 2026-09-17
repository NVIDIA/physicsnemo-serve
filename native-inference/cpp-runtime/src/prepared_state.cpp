#include "physicsnemo/inference/prepared_state.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace physicsnemo::inference {

SharedTensor::SharedTensor(std::string name, DType dtype, Device device,
                           Shape shape, const void* data,
                           std::size_t byte_size, std::shared_ptr<void> owner)
    : name_(std::move(name)),
      dtype_(dtype),
      device_(device),
      shape_(std::move(shape)),
      data_(data),
      byte_size_(byte_size),
      owner_(std::move(owner)) {
  const std::size_t expected = element_count(shape_) * dtype_size(dtype_);
  if (byte_size_ != expected || (expected != 0 && data_ == nullptr)) {
    throw std::invalid_argument(
        "shared tensor storage does not match shape");
  }
  if (expected != 0 && owner_ == nullptr) {
    throw std::invalid_argument("shared tensor requires an allocation owner");
  }
  if (device_.index < 0) {
    throw std::invalid_argument("shared tensor device index cannot be negative");
  }
}

TensorView SharedTensor::view() const {
  return {name_, dtype_, device_, shape_, data_, byte_size_};
}

PreparedState::PreparedState(std::string key,
                             std::vector<SharedTensor> tensors)
    : key_(std::move(key)), tensors_(std::move(tensors)) {
  if (key_.empty()) {
    throw std::invalid_argument("prepared state key cannot be empty");
  }
  std::unordered_set<std::string> names;
  for (const auto& tensor : tensors_) {
    if (!names.insert(tensor.view().name).second) {
      throw std::invalid_argument(
          "prepared state tensor names must be unique");
    }
  }
}

std::vector<TensorView> PreparedState::views() const {
  std::vector<TensorView> result;
  result.reserve(tensors_.size());
  for (const auto& tensor : tensors_) result.push_back(tensor.view());
  return result;
}

std::size_t PreparedState::byte_size() const {
  std::size_t result = 0;
  for (const auto& tensor : tensors_) result += tensor.byte_size();
  return result;
}

}  // namespace physicsnemo::inference
