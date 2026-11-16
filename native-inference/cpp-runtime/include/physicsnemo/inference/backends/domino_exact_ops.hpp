/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace physicsnemo::inference::domino {

inline constexpr std::string_view kExactOperatorId{
    "physicsnemo-cfd.domino-exact-boundary"};
inline constexpr std::string_view kExactOperatorAbi{
    "b1c60ddada2438469a1d24b4e53ae196425b73648f6d8ae45ecf64043755d7e6"};

void register_exact_ops();

}  // namespace physicsnemo::inference::domino
