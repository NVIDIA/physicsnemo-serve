#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactAttentionOperatorId =
    "pnmir.tensorrt-exact-attention";
inline constexpr std::string_view kTensorRTExactAttentionOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_attention_register() noexcept;
