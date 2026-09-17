#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactWeightedBlendOperatorId =
    "pnmir.tensorrt-exact-weighted-blend";
inline constexpr std::string_view kTensorRTExactWeightedBlendOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_weighted_blend_register() noexcept;
