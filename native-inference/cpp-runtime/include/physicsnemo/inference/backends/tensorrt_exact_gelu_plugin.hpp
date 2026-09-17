#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactGeluOperatorId =
    "pnmir.tensorrt-exact-gelu";
inline constexpr std::string_view kTensorRTExactGeluOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_gelu_register() noexcept;
