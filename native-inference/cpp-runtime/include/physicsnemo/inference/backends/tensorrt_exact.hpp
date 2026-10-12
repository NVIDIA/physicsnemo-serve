#pragma once

#include <string_view>

namespace physicsnemo::inference {

inline constexpr std::string_view kTensorRTExactAttentionOperatorId =
    "pnmir.tensorrt-exact-attention";
inline constexpr std::string_view kTensorRTExactAttentionOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactDesliceBmmOperatorId =
    "pnmir.tensorrt-exact-deslice-bmm";
inline constexpr std::string_view kTensorRTExactDesliceBmmOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactGeluOperatorId =
    "pnmir.tensorrt-exact-gelu";
inline constexpr std::string_view kTensorRTExactGeluOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactGemmOperatorId =
    "pnmir.tensorrt-exact-gemm";
inline constexpr std::string_view kTensorRTExactGemmOperatorAbi = "1";

inline constexpr std::string_view
    kTensorRTExactInverseDistanceBlendOperatorId =
        "pnmir.tensorrt-exact-inverse-distance-blend";
inline constexpr std::string_view
    kTensorRTExactInverseDistanceBlendOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactLayerNormOperatorId =
    "pnmir.tensorrt-exact-layer-norm";
inline constexpr std::string_view kTensorRTExactLayerNormOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactLinearOperatorId =
    "pnmir.tensorrt-exact-linear";
inline constexpr std::string_view kTensorRTExactLinearOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactScalarDivOperatorId =
    "pnmir.tensorrt-exact-scalar-div";
inline constexpr std::string_view kTensorRTExactScalarDivOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactSliceBmmOperatorId =
    "pnmir.tensorrt-exact-slice-bmm";
inline constexpr std::string_view kTensorRTExactSliceBmmOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactSoftmaxOperatorId =
    "pnmir.tensorrt-exact-softmax";
inline constexpr std::string_view kTensorRTExactSoftmaxOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactTokenSumOperatorId =
    "pnmir.tensorrt-exact-token-sum";
inline constexpr std::string_view kTensorRTExactTokenSumOperatorAbi = "1";

inline constexpr std::string_view kTensorRTExactWeightedBlendOperatorId =
    "pnmir.tensorrt-exact-weighted-blend";
inline constexpr std::string_view kTensorRTExactWeightedBlendOperatorAbi = "1";

}  // namespace physicsnemo::inference

extern "C" int pnmir_tensorrt_exact_attention_register() noexcept;
extern "C" int pnmir_tensorrt_exact_deslice_bmm_register() noexcept;
extern "C" int pnmir_tensorrt_exact_gelu_register() noexcept;
extern "C" int pnmir_tensorrt_exact_gemm_register() noexcept;
extern "C" int pnmir_tensorrt_exact_inverse_distance_blend_register() noexcept;
extern "C" int pnmir_tensorrt_exact_layer_norm_register() noexcept;
extern "C" int pnmir_tensorrt_exact_linear_register() noexcept;
extern "C" int pnmir_tensorrt_exact_scalar_div_register() noexcept;
extern "C" int pnmir_tensorrt_exact_slice_bmm_register() noexcept;
extern "C" int pnmir_tensorrt_exact_softmax_register() noexcept;
extern "C" int pnmir_tensorrt_exact_token_sum_register() noexcept;
extern "C" int pnmir_tensorrt_exact_weighted_blend_register() noexcept;
