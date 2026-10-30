#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>

namespace physicsnemo::inference::v1 {
struct SelfTestComparison {
  double max_absolute_error{0};
  double max_relative_error{0};
};
// Symmetric tolerances, with no NaN/Inf or empty-fixture acceptance. Expected
// values must come from an independent authored/export reference fixture.
inline SelfTestComparison compare_self_test_float32(std::span<const float> actual,
    std::span<const float> expected, double rtol, double atol) {
  if (actual.empty() || actual.size() != expected.size())
    throw std::invalid_argument("self_test element count differs or is empty");
  if (!std::isfinite(rtol) || rtol < 0 || !std::isfinite(atol) || atol < 0)
    throw std::invalid_argument("self_test tolerances must be finite and nonnegative");
  SelfTestComparison report;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
      throw std::invalid_argument("self_test actual and expected values must be finite");
    const double left = actual[index], right = expected[index];
    const double error = std::abs(left - right);
    const double scale = std::max(std::abs(left), std::abs(right));
    if (error > atol + rtol * scale)
      throw std::invalid_argument("startup self_test output mismatch at element " + std::to_string(index));
    report.max_absolute_error = std::max(report.max_absolute_error, error);
    report.max_relative_error = std::max(report.max_relative_error, scale == 0 ? 0 : error / scale);
  }
  return report;
}
}
