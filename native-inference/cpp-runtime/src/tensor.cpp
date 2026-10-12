#include "physicsnemo/inference/tensor.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace physicsnemo::inference {

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kInt32:
      return 4;
    case DType::kInt64:
      return 8;
    case DType::kUInt8:
      return 1;
  }
  throw std::invalid_argument("unsupported dtype");
}

std::string_view to_string(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kUInt8:
      return "uint8";
  }
  throw std::invalid_argument("unsupported dtype");
}

std::string_view to_string(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::kCpu:
      return "cpu";
    case DeviceType::kCuda:
      return "cuda";
  }
  throw std::invalid_argument("unsupported device type");
}

DType parse_dtype(std::string_view value) {
  if (value == "float32") return DType::kFloat32;
  if (value == "float16") return DType::kFloat16;
  if (value == "bfloat16") return DType::kBFloat16;
  if (value == "int32") return DType::kInt32;
  if (value == "int64") return DType::kInt64;
  if (value == "uint8") return DType::kUInt8;
  throw std::invalid_argument("unknown dtype: " + std::string(value));
}

DeviceType parse_device_type(std::string_view value) {
  if (value == "cpu") return DeviceType::kCpu;
  if (value == "cuda") return DeviceType::kCuda;
  throw std::invalid_argument("unknown device type: " + std::string(value));
}

std::size_t element_count(const Shape& shape) {
  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument("tensor dimensions must be positive");
    }
    const auto size = static_cast<std::size_t>(dimension);
    if (count > std::numeric_limits<std::size_t>::max() / size) {
      throw std::overflow_error("tensor element count overflow");
    }
    count *= size;
  }
  return count;
}

std::size_t tensor_byte_size(const Shape& shape, DType dtype) {
  const auto count = element_count(shape);
  const auto width = dtype_size(dtype);
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("tensor byte size overflow");
  }
  return count * width;
}

OwnedTensor::OwnedTensor(std::string name, DType dtype, Device device,
                         Shape shape, std::vector<std::byte> storage)
    : name_(std::move(name)),
      dtype_(dtype),
      device_(device),
      shape_(std::move(shape)),
      storage_(std::move(storage)) {
  const std::size_t expected = tensor_byte_size(shape_, dtype_);
  if (storage_.size() != expected) {
    throw std::invalid_argument("owned tensor storage size does not match shape");
  }
}

TensorView OwnedTensor::view() const {
  return TensorView{name_, dtype_, device_, shape_, storage_.data(),
                    storage_.size()};
}

}  // namespace physicsnemo::inference
