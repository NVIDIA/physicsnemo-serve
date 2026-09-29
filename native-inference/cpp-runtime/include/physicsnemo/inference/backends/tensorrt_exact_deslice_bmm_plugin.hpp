#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactDesliceBmmOperatorId =
    "pnmir.tensorrt-exact-deslice-bmm";
inline constexpr std::string_view kTensorRTExactDesliceBmmOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_deslice_bmm_register() noexcept;
