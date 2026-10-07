#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace device_runner {

inline std::size_t parse_count(const char* value, const std::string& name,
                              bool allow_zero) {
  const std::string text(value);
  std::size_t parsed = 0;
  const unsigned long long result = std::stoull(text, &parsed);
  // stoull accepts a minus sign and wraps the result to unsigned.
  if (text.find('-') != std::string::npos || parsed != text.size() ||
      (!allow_zero && result == 0)) {
    throw std::invalid_argument(name + " must be a valid count");
  }
  return static_cast<std::size_t>(result);
}

}  // namespace device_runner
