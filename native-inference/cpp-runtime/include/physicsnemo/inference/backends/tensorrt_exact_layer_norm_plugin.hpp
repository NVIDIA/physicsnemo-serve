#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactLayerNormOperatorId =
    "pnmir.tensorrt-exact-layer-norm";
inline constexpr std::string_view kTensorRTExactLayerNormOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_layer_norm_register() noexcept;
