#pragma once

#include <memory>

#include "physicsnemo/inference/backend.hpp"

namespace physicsnemo::inference {

class Runtime;

std::unique_ptr<Backend> create_aoti_backend();

// Register the DoMINO sidecar and ABI before loading a package that requires it.
// A build without PNMIR_BUILD_DOMINO_EXACT_OPS leaves the registry unchanged.
void register_aoti_exact_operators(Runtime& runtime);

}  // namespace physicsnemo::inference
