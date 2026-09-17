#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactGemmOperatorId =
    "pnmir.tensorrt-exact-gemm";
inline constexpr std::string_view kTensorRTExactGemmOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_gemm_register() noexcept;
