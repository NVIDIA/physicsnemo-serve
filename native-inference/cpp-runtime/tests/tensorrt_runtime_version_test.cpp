#include <iostream>
#include <string_view>

#include "../src/tensorrt/runtime_version.hpp"

int main() {
  using physicsnemo::inference::detail::tensorrt_runtime_version_matches;
  constexpr std::string_view runtime = "10.14.1.48";
  struct Case {
    std::string_view artifact;
    bool expected;
  };
  constexpr Case cases[] = {
      {"", true},
      {"10.14.1.48", true},
      {"10.14.1.48.post1", true},
      {"10.14.1.48.post0", true},
      {"10.14.1.48.post42", true},
      {"10.14.1.49.post1", false},
      {"10.14.2.48.post1", false},
      {"10.15.1.48.post1", false},
      {"11.14.1.48.post1", false},
      {"10.14.1", false},
      {"10.14.1.49", false},
      {"10.14.2.48", false},
      {"10.15.1.48", false},
      {"11.14.1.48", false},
      {"10.14.1.480", false},
      {"10.14.1.048", false},
      {"010.14.1.48", false},
      {"10.14.1.48.post", false},
      {"10.14.1.48.post-1", false},
      {"10.14.1.48.post+1", false},
      {"10.14.1.48.post1x", false},
      {"10.14.1.48.post1.post2", false},
      {"10.14.1.48.post1+cu13", false},
      {"10.14.1.48.post1\n", false},
      {"10.14.1.48.dev1", false},
      {"10.14.1.48.rc1", false},
      {"10.14.1.48+cu13", false},
      {" 10.14.1.48", false},
      {"10.14.1.48 ", false},
  };
  int failures = 0;
  for (const auto& test : cases) {
    if (tensorrt_runtime_version_matches(test.artifact, runtime) != test.expected) {
      std::cerr << "Unexpected TensorRT compatibility for '" << test.artifact
                << "' against '" << runtime << "'\n";
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
