#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace transolver_workflow {
// Dynamic packages are tested at the configured full block size at startup.
inline std::vector<std::int64_t> startup_fixture_points(
    const std::vector<std::int64_t>& package_points, std::int64_t block_size) {
  std::vector<std::int64_t> result;
  for (const auto points : package_points) {
    const auto effective = points == -1 ? block_size : points;
    if (std::ranges::find(result, effective) == result.end())
      result.push_back(effective);
  }
  return result;
}

inline void validate_startup_output(const std::vector<std::int64_t>& actual_shape,
                                    const std::vector<std::int64_t>& expected_shape,
                                    std::span<const float> values) {
  if (actual_shape != expected_shape)
    throw std::invalid_argument("startup output shape differs from declared model");
  std::size_t count = 1;
  for (const auto dimension : expected_shape) {
    if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
          std::numeric_limits<std::size_t>::max() / count)
      throw std::invalid_argument("invalid startup output shape");
    count *= static_cast<std::size_t>(dimension);
  }
  if (values.size() != count)
    throw std::invalid_argument("startup output element count differs");
  if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); }))
    throw std::invalid_argument("startup output must be finite");
}
}
