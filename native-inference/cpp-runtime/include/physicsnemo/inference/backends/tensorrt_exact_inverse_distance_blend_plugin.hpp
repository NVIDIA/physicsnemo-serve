#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view
    kTensorRTExactInverseDistanceBlendOperatorId =
        "pnmir.tensorrt-exact-inverse-distance-blend";
inline constexpr std::string_view
    kTensorRTExactInverseDistanceBlendOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int
pnmir_tensorrt_exact_inverse_distance_blend_register() noexcept;
