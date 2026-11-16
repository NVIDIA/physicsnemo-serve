/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "physicsnemo/inference/backend.hpp"

namespace physicsnemo::inference {

std::unique_ptr<Backend> create_onnxruntime_backend();

}  // namespace physicsnemo::inference
