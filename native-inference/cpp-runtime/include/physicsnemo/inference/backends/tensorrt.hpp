#pragma once

#include <memory>

#include "physicsnemo/inference/backend.hpp"

namespace physicsnemo::inference {

class Runtime;

std::unique_ptr<Backend> create_tensorrt_backend();

// Register the operator IDs provided by a PNMIR_ENABLE_TENSORRT_EXACT build.
// Call before create_session for packages declaring these required operators.
// A TensorRT build without exact operators leaves the registry unchanged.
void register_tensorrt_exact_operators(Runtime& runtime);

}  // namespace physicsnemo::inference
