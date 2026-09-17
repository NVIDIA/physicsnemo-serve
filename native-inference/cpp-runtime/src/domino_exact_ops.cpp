#include "physicsnemo/inference/backends/domino_exact_ops.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <ATen/Functions.h>
#include <torch/library.h>

namespace physicsnemo::inference::domino {
namespace {

at::Tensor nearest_upsample3d_2x(const at::Tensor& input) {
  const std::array<double, 3> scale_factors{2.0, 2.0, 2.0};
  return at::upsample_nearest3d(
      input, std::nullopt, at::ArrayRef<double>(scale_factors));
}

at::Tensor sdf_features(const at::Tensor& sdf_grid,
                        const at::Tensor& scaling_factors) {
  TORCH_CHECK(sdf_grid.dim() == 4,
              "DoMINO SDF grid must have shape [batch, nx, ny, nz]");
  TORCH_CHECK(scaling_factors.dim() == 1 && scaling_factors.numel() > 0,
              "DoMINO SDF scaling factors must be a non-empty vector");
  TORCH_CHECK(sdf_grid.device() == scaling_factors.device(),
              "DoMINO SDF grid and scaling factors must share a device");
  TORCH_CHECK(sdf_grid.scalar_type() == scaling_factors.scalar_type(),
              "DoMINO SDF grid and scaling factors must share a dtype");

  const auto sdf = sdf_grid.unsqueeze(1);
  const auto binary_sdf = at::where(sdf.ge(0.0), 0.0, 1.0);
  const auto gradients = at::gradient(sdf, at::IntArrayRef{2, 3, 4});

  std::vector<at::Tensor> features;
  features.reserve(static_cast<std::size_t>(5 + scaling_factors.numel()));
  features.push_back(sdf);
  const auto absolute_sdf = sdf.abs();
  for (std::int64_t index = 0; index < scaling_factors.numel(); ++index) {
    features.push_back(
        at::div(sdf, at::add(scaling_factors[index], absolute_sdf)));
  }
  features.push_back(binary_sdf);
  features.insert(features.end(), gradients.begin(), gradients.end());
  return at::cat(features, 1);
}

void check_scalar_tensor(const at::Tensor& value, const at::Tensor& scalar) {
  TORCH_CHECK(scalar.dim() == 0,
              "exact scalar arithmetic requires a zero-dimensional tensor");
  TORCH_CHECK(value.device() == scalar.device(),
              "exact scalar operands must share a device");
  TORCH_CHECK(value.scalar_type() == scalar.scalar_type(),
              "exact scalar operands must share a dtype");
}

at::Tensor tensor_scalar_add(const at::Tensor& value,
                             const at::Tensor& scalar) {
  check_scalar_tensor(value, scalar);
  return at::add(value, scalar.item());
}

at::Tensor tensor_scalar_div(const at::Tensor& value,
                             const at::Tensor& scalar) {
  check_scalar_tensor(value, scalar);
  return at::div(value, scalar.item());
}

at::Tensor tensor_scalar_mul(const at::Tensor& value,
                             const at::Tensor& scalar) {
  check_scalar_tensor(value, scalar);
  return at::mul(value, scalar.item());
}

at::Tensor reciprocal(const at::Tensor& value) {
  return at::reciprocal(value);
}

at::Tensor index_select_dim0(const at::Tensor& value,
                             const at::Tensor& index) {
  return at::index_select(value, 0, index.to(at::kLong));
}

at::Tensor index_select_dim1(const at::Tensor& value,
                             const at::Tensor& index) {
  return at::index_select(value, 1, index.to(at::kLong));
}

at::Tensor tensor_sub(const at::Tensor& left, const at::Tensor& right) {
  return at::sub(left, right);
}

at::Tensor vector_norm_last_dim(const at::Tensor& value) {
  return at::linalg_vector_norm(
      value, 2.0, at::IntArrayRef{-1}, true, std::nullopt);
}

}  // namespace

TORCH_LIBRARY(pnmir_domino, library) {
  library.def("nearest_upsample3d_2x(Tensor input) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("sdf_features(Tensor sdf_grid, Tensor scaling_factors) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("tensor_scalar_add(Tensor value, Tensor scalar) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("tensor_scalar_div(Tensor value, Tensor scalar) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("tensor_scalar_mul(Tensor value, Tensor scalar) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("reciprocal(Tensor value) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("index_select_dim0(Tensor value, Tensor index) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("index_select_dim1(Tensor value, Tensor index) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("tensor_sub(Tensor left, Tensor right) -> Tensor",
              {at::Tag::pt2_compliant_tag});
  library.def("vector_norm_last_dim(Tensor value) -> Tensor",
              {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(pnmir_domino, CPU, library) {
  library.impl("nearest_upsample3d_2x", &nearest_upsample3d_2x);
  library.impl("sdf_features", &sdf_features);
  library.impl("tensor_scalar_add", &tensor_scalar_add);
  library.impl("tensor_scalar_div", &tensor_scalar_div);
  library.impl("tensor_scalar_mul", &tensor_scalar_mul);
  library.impl("reciprocal", &reciprocal);
  library.impl("index_select_dim0", &index_select_dim0);
  library.impl("index_select_dim1", &index_select_dim1);
  library.impl("tensor_sub", &tensor_sub);
  library.impl("vector_norm_last_dim", &vector_norm_last_dim);
}

TORCH_LIBRARY_IMPL(pnmir_domino, CUDA, library) {
  library.impl("nearest_upsample3d_2x", &nearest_upsample3d_2x);
  library.impl("sdf_features", &sdf_features);
  library.impl("tensor_scalar_add", &tensor_scalar_add);
  library.impl("tensor_scalar_div", &tensor_scalar_div);
  library.impl("tensor_scalar_mul", &tensor_scalar_mul);
  library.impl("reciprocal", &reciprocal);
  library.impl("index_select_dim0", &index_select_dim0);
  library.impl("index_select_dim1", &index_select_dim1);
  library.impl("tensor_sub", &tensor_sub);
  library.impl("vector_norm_last_dim", &vector_norm_last_dim);
}

void register_exact_ops() {
  // Referencing this symbol keeps the shared library linked. Dispatcher
  // registration is performed by the TORCH_LIBRARY static initializers.
}

}  // namespace physicsnemo::inference::domino
