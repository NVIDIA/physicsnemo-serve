#pragma once

#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

struct ExecutionContext {
  Device device{};
  void* stream{nullptr};
};

}  // namespace physicsnemo::inference
