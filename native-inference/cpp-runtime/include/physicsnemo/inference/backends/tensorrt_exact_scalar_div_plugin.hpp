#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactScalarDivOperatorId =
    "pnmir.tensorrt-exact-scalar-div";
inline constexpr std::string_view kTensorRTExactScalarDivOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_scalar_div_register() noexcept;
