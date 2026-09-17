#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactLinearOperatorId =
    "pnmir.tensorrt-exact-linear";
inline constexpr std::string_view kTensorRTExactLinearOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_linear_register() noexcept;
