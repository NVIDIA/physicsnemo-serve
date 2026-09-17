#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactSoftmaxOperatorId =
    "pnmir.tensorrt-exact-softmax";
inline constexpr std::string_view kTensorRTExactSoftmaxOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_softmax_register() noexcept;
