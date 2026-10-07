#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

struct TensorSpec {
  std::string name;
  DType dtype{DType::kFloat32};
  Shape shape;
};

struct CustomOperatorSpec {
  std::string id;
  std::string abi;
};

struct ArtifactSpec {
  std::string backend;
  DeviceType target{DeviceType::kCpu};
  std::string precision;
  std::string runtime_version;
  std::filesystem::path path;
  std::vector<CustomOperatorSpec> required_operators;
};

struct ModelManifest {
  int format_version{0};
  std::string model_name;
  std::string model_version;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::vector<ArtifactSpec> artifacts;
};

class ModelPackage {
 public:
  static ModelPackage load(const std::filesystem::path& root);

  const std::filesystem::path& root() const { return root_; }
  const ModelManifest& manifest() const { return manifest_; }
  std::filesystem::path artifact_path(const ArtifactSpec& artifact) const;

 private:
  std::filesystem::path root_;
  ModelManifest manifest_;
};

}  // namespace physicsnemo::inference
