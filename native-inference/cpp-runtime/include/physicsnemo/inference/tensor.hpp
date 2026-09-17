#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace physicsnemo::inference {

enum class DType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kInt32,
  kInt64,
  kUInt8,
};

enum class DeviceType {
  kCpu,
  kCuda,
};

struct Device {
  DeviceType type{DeviceType::kCpu};
  int index{0};

  bool operator==(const Device&) const = default;
};

using Shape = std::vector<std::int64_t>;

std::size_t dtype_size(DType dtype);
std::string_view to_string(DType dtype);
std::string_view to_string(DeviceType device_type);
DType parse_dtype(std::string_view value);
DeviceType parse_device_type(std::string_view value);
std::size_t element_count(const Shape& shape);

struct TensorView {
  std::string name;
  DType dtype{DType::kFloat32};
  Device device{};
  Shape shape;
  const void* data{nullptr};
  std::size_t byte_size{0};
};

struct MutableTensorView {
  std::string name;
  DType dtype{DType::kFloat32};
  Device device{};
  Shape shape;
  void* data{nullptr};
  std::size_t byte_size{0};

  TensorView as_read_only() const {
    return {name, dtype, device, shape, data, byte_size};
  }
};

class OwnedTensor {
 public:
  OwnedTensor(std::string name, DType dtype, Device device, Shape shape,
              std::vector<std::byte> storage);

  TensorView view() const;
  const std::vector<std::byte>& storage() const { return storage_; }

 private:
  std::string name_;
  DType dtype_;
  Device device_;
  Shape shape_;
  std::vector<std::byte> storage_;
};

}  // namespace physicsnemo::inference
