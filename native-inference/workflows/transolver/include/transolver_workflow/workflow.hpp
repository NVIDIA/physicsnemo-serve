#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <torch/torch.h>

#include "physicsnemo/inference/manifest.hpp"

namespace transolver_workflow {

enum class Domain {
  kSurface,
  kVolume,
};

struct PreparedInputs {
  torch::Tensor fx;
  torch::Tensor embedding;
  float volume_length_scale{0.0F};
  std::string mesh_reader;
};

struct NormalizationStats {
  torch::Tensor mean;
  torch::Tensor standard_deviation;
};

Domain parse_domain(std::string_view value);
std::string_view to_string(Domain domain);
std::int64_t embedding_width(Domain domain);
std::int64_t output_width(Domain domain);

// Reads only the inline binary Float32 Points payload from a DrivAerML VTU.
// A non-zero point_limit bounds both file I/O and host/device allocation.
torch::Tensor read_inline_binary_vtu_points(
    const std::filesystem::path& path,
    std::int64_t point_limit,
    const torch::Device& device);

// Sequential bounded reader for very large inline-binary VTU point arrays.
// Calls advance one decoder, so earlier points are neither reparsed nor kept.
class InlineBinaryVtuPointStream {
 public:
  explicit InlineBinaryVtuPointStream(const std::filesystem::path& path);
  ~InlineBinaryVtuPointStream();

  InlineBinaryVtuPointStream(InlineBinaryVtuPointStream&&) noexcept;
  InlineBinaryVtuPointStream& operator=(InlineBinaryVtuPointStream&&) noexcept;
  InlineBinaryVtuPointStream(const InlineBinaryVtuPointStream&) = delete;
  InlineBinaryVtuPointStream& operator=(const InlineBinaryVtuPointStream&) = delete;

  std::int64_t total_points() const;
  std::int64_t point_offset() const;
  torch::Tensor read_next(std::int64_t point_count,
                          const torch::Device& device);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

PreparedInputs prepare_inputs(
    const std::filesystem::path& mesh_path,
    const std::filesystem::path& stl_path,
    Domain domain,
    const torch::Device& device,
    float air_density,
    float stream_velocity,
    std::int64_t point_limit = 0);

NormalizationStats load_normalization_stats(
    const std::filesystem::path& path,
    Domain domain,
    const torch::Device& device);

torch::Tensor decode_physical_outputs(
    const torch::Tensor& standardized,
    const NormalizationStats& stats,
    Domain domain,
    double air_density,
    double stream_velocity,
    double volume_length_scale);

void validate_package_contract(
    const physicsnemo::inference::ModelManifest& manifest,
    Domain domain,
    std::int64_t block_size);

}  // namespace transolver_workflow
