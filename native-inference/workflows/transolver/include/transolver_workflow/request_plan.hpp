#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace transolver_workflow {

struct InferenceBlock {
  std::int64_t begin;
  std::int64_t points;
  bool operator==(const InferenceBlock&) const = default;
};

// Preserve the exact point grouping. Never pad or merge a tail block.
std::vector<InferenceBlock> plan_blocks(std::int64_t point_count, std::int64_t block_size);
// -1 represents a deliberately qualified dynamic package.
void preflight_blocks(const std::vector<InferenceBlock>& blocks,
                      const std::vector<std::int64_t>& package_points);
// PhysicsNeMo-CFD 921f14dc per-case seed: sha256(str(seed) + NUL + case_id),
// first eight digest bytes little-endian, with the sign bit cleared.
std::int64_t provider_case_seed(std::int64_t run_seed, std::string_view case_id);

}  // namespace transolver_workflow
