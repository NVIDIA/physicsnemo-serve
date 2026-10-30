#pragma once

#include <string_view>

namespace physicsnemo::inference::detail {

inline bool tensorrt_runtime_version_matches(std::string_view artifact,
                                            std::string_view runtime) {
  if (artifact.empty() || artifact == runtime) return true;
  if (!artifact.starts_with(runtime)) return false;
  auto suffix = artifact.substr(runtime.size());
  // Python distributions can append a post-release packaging revision while
  // retaining the exact four-component native TensorRT runtime version.
  constexpr std::string_view post_release = ".post";
  if (!suffix.starts_with(post_release)) return false;
  suffix.remove_prefix(post_release.size());
  if (suffix.empty()) return false;
  for (const char digit : suffix) {
    if (digit < '0' || digit > '9') return false;
  }
  return true;
}

}  // namespace physicsnemo::inference::detail
