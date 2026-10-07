#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../app/device_runner_count.hpp"

namespace {

int check_count(const std::string& text, const std::string& name,
                bool allow_zero, std::size_t expected) {
  try {
    const auto result = device_runner::parse_count(text.c_str(), name, allow_zero);
    if (result == expected) return 0;
    std::cerr << name << ' ' << std::quoted(text) << " returned " << result
              << ", expected " << expected << '\n';
  } catch (const std::exception& error) {
    std::cerr << name << ' ' << std::quoted(text)
              << " unexpectedly rejected: " << error.what() << '\n';
  }
  return 1;
}

int check_rejected(const std::string& text, const std::string& name,
                   bool allow_zero) {
  try {
    const auto result = device_runner::parse_count(text.c_str(), name, allow_zero);
    std::cerr << name << ' ' << std::quoted(text)
              << " was accepted as " << result << "; expected rejection\n";
  } catch (const std::invalid_argument&) {
    return 0;
  } catch (const std::out_of_range&) {
    return 0;
  }
  return 1;
}

}  // namespace

int main() {
  int failures = 0;
  for (const bool allow_zero : {true, false}) {
    const std::string name = allow_zero ? "warmup" : "iterations";
    for (const char* text : {"-1", " -1", "\t-1", "\n-2", "-0"}) {
      failures += check_rejected(text, name, allow_zero);
    }
    failures += check_count("1", name, allow_zero, 1);
    failures += check_count("42", name, allow_zero, 42);
    failures += check_count("+7", name, allow_zero, 7);
    failures += check_count(" \t+7", name, allow_zero, 7);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    failures += check_count(std::to_string(maximum), name, allow_zero, maximum);
    for (const char* text : {"", "invalid", "1x", "1 "}) {
      failures += check_rejected(text, name, allow_zero);
    }
    failures += check_rejected(
        std::to_string(std::numeric_limits<unsigned long long>::max()) + "0",
        name, allow_zero);
  }
  failures += check_count("0", "warmup", true, 0);
  failures += check_rejected("0", "iterations", false);
  if (failures != 0) return 1;
  std::cout << "Device runner count parsing passed\n";
  return 0;
}
