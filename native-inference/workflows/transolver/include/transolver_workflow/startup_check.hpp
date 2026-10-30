#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace transolver_workflow {
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
