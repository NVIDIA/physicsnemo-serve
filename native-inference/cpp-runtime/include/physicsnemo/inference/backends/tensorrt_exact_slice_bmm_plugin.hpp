#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactSliceBmmOperatorId =
    "pnmir.tensorrt-exact-slice-bmm";
inline constexpr std::string_view kTensorRTExactSliceBmmOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_slice_bmm_register() noexcept;
