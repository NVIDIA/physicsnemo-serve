#include "transolver_workflow/request_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace transolver_workflow {

std::vector<InferenceBlock> plan_blocks(const std::int64_t point_count,
                                       const std::int64_t block_size) {
  if (block_size < 2) {
    throw std::invalid_argument("block_size must be at least two");
  }
  if (point_count < 2) {
    throw std::invalid_argument("Transolver requires at least two mesh locations");
  }
  if (point_count % block_size == 1) {
    throw std::invalid_argument(
        "the final block contains one point, outside the qualified dynamic package contract");
  }
  std::vector<InferenceBlock> blocks;
  for (std::int64_t begin = 0; begin < point_count;) {
    const auto points = std::min(block_size, point_count - begin);
    blocks.push_back({begin, points});
    begin += points;
  }
  return blocks;
}

void preflight_blocks(const std::vector<InferenceBlock>& blocks,
                      const std::vector<std::int64_t>& package_points) {
  for (const auto& block : blocks) {
    if (std::ranges::find(package_points, block.points) == package_points.end() &&
        std::ranges::find(package_points, -1) == package_points.end()) {
      throw std::invalid_argument("no package accepts a " +
                                  std::to_string(block.points) + "-point block");
    }
  }
}

std::int64_t provider_case_seed(const std::int64_t run_seed,
                                const std::string_view case_id) {
  // SHA-256 is kept here to make the request contract independent of CUDA,
  // Torch and platform crypto libraries. This is a seed, not authentication.
  constexpr std::array<std::uint32_t, 64> k{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::array<std::uint32_t, 8> h{
      0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto text = std::to_string(run_seed);
  text.push_back('\0');
  text.append(case_id);
  std::vector<std::uint8_t> bytes(text.begin(), text.end());
  const auto bits = static_cast<std::uint64_t>(bytes.size()) * 8U;
  bytes.push_back(0x80);
  while (bytes.size() % 64U != 56U) bytes.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(bits >> shift);
  for (std::size_t base = 0; base < bytes.size(); base += 64U) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16U; ++i) {
      for (std::size_t j = 0; j < 4U; ++j) w[i] = (w[i] << 8U) | bytes[base + i * 4U + j];
    }
    for (std::size_t i = 16; i < 64U; ++i) {
      const auto s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], v=h[7];
    for (std::size_t i = 0; i < 64U; ++i) {
      const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto t1 = v + s1 + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
      v=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=v;
  }
  std::uint64_t seed = 0;
  for (std::size_t i = 0; i < 8U; ++i) {
    const auto byte = (h[i / 4U] >> (24U - 8U * (i % 4U))) & 0xffU;
    seed |= static_cast<std::uint64_t>(byte) << (8U * i);
  }
  return static_cast<std::int64_t>(seed & 0x7fffffffffffffffULL);
}

}  // namespace transolver_workflow
