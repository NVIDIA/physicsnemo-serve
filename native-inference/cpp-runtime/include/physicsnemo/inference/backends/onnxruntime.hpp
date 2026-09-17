#pragma once

#include <memory>

#include "physicsnemo/inference/backend.hpp"

namespace physicsnemo::inference {

std::unique_ptr<Backend> create_onnxruntime_backend();

}  // namespace physicsnemo::inference
