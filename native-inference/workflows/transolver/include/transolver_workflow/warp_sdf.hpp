#pragma once

#include <torch/torch.h>

namespace transolver_workflow {

struct SignedDistanceResult {
  torch::Tensor distance;
  torch::Tensor closest_point;
};

SignedDistanceResult warp_signed_distance(
    const torch::Tensor& mesh_vertices,
    const torch::Tensor& mesh_indices,
    const torch::Tensor& query_points);

}  // namespace transolver_workflow
