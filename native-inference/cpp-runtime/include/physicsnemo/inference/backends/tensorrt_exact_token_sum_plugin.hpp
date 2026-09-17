#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactTokenSumOperatorId =
    "pnmir.tensorrt-exact-token-sum";
inline constexpr std::string_view kTensorRTExactTokenSumOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_token_sum_register() noexcept;
