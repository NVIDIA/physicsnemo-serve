#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

class SharedTensor {
 public:
  // `owner` controls allocation lifetime; `data` may point into that
  // allocation. Callers must not resize or mutate the allocation metadata.
  SharedTensor(std::string name, DType dtype, Device device, Shape shape,
               const void* data, std::size_t byte_size,
               std::shared_ptr<void> owner);

  TensorView view() const;
  std::size_t byte_size() const { return byte_size_; }

 private:
  std::string name_;
  DType dtype_;
  Device device_;
  Shape shape_;
  const void* data_;
  std::size_t byte_size_;
  std::shared_ptr<void> owner_;
};

class PreparedState {
 public:
  // Prepared state is immutable after construction. Domain adapters own cache
  // identity and invalidation policy; this class owns only named tensors.
  PreparedState(std::string key, std::vector<SharedTensor> tensors);

  const std::string& key() const { return key_; }
  const std::vector<SharedTensor>& tensors() const { return tensors_; }
  std::vector<TensorView> views() const;
  std::size_t byte_size() const;

 private:
  std::string key_;
  std::vector<SharedTensor> tensors_;
};

}  // namespace physicsnemo::inference
