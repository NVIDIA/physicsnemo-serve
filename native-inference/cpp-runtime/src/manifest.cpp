#include "physicsnemo/inference/manifest.hpp"

#include <fstream>
#include <stdexcept>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace physicsnemo::inference {
namespace {

using Json = nlohmann::json;

TensorSpec parse_tensor_spec(const Json& value) {
  TensorSpec spec;
  spec.name = value.at("name").get<std::string>();
  spec.dtype = parse_dtype(value.at("dtype").get<std::string>());
  spec.shape = value.at("shape").get<Shape>();
  if (spec.name.empty()) {
    throw std::invalid_argument("tensor name cannot be empty");
  }
  for (const auto dimension : spec.shape) {
    if (dimension == 0 || dimension < -1) {
      throw std::invalid_argument("invalid dimension in tensor: " + spec.name);
    }
  }
  return spec;
}

void ensure_unique_names(const std::vector<TensorSpec>& specs,
                         const std::string& kind) {
  std::unordered_set<std::string> names;
  for (const auto& spec : specs) {
    if (!names.insert(spec.name).second) {
      throw std::invalid_argument("duplicate " + kind + " tensor: " +
                                  spec.name);
    }
  }
}

bool safe_relative_path(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) return false;
  for (const auto& component : path) {
    if (component == "..") return false;
  }
  return true;
}

bool path_is_within(const std::filesystem::path& path,
                    const std::filesystem::path& root) {
  auto path_component = path.begin();
  auto root_component = root.begin();
  for (; root_component != root.end(); ++root_component, ++path_component) {
    if (path_component == path.end() || *path_component != *root_component) {
      return false;
    }
  }
  return true;
}

std::vector<CustomOperatorSpec> parse_required_operators(const Json& value) {
  std::vector<CustomOperatorSpec> operators;
  std::unordered_set<std::string> ids;
  for (const auto& entry : value) {
    CustomOperatorSpec spec{
        entry.at("id").get<std::string>(),
        entry.at("abi").get<std::string>(),
    };
    if (spec.id.empty() || spec.abi.empty()) {
      throw std::invalid_argument(
          "required operator id and ABI cannot be empty");
    }
    if (!ids.insert(spec.id).second) {
      throw std::invalid_argument("duplicate required operator: " + spec.id);
    }
    operators.push_back(std::move(spec));
  }
  return operators;
}

}  // namespace

ModelPackage ModelPackage::load(const std::filesystem::path& root) {
  if (!std::filesystem::is_directory(root)) {
    throw std::invalid_argument("package root is not a directory: " +
                                root.string());
  }

  const auto canonical_root = std::filesystem::canonical(root);
  const auto manifest_path = canonical_root / "model.json";
  std::ifstream stream(manifest_path);
  if (!stream) {
    throw std::invalid_argument("cannot open package manifest: " +
                                manifest_path.string());
  }

  Json document;
  stream >> document;

  ModelManifest manifest;
  manifest.format_version = document.at("format_version").get<int>();
  if (manifest.format_version != 1) {
    throw std::invalid_argument("unsupported package format version");
  }

  const auto& model = document.at("model");
  manifest.model_name = model.at("name").get<std::string>();
  manifest.model_version = model.at("version").get<std::string>();
  if (manifest.model_name.empty() || manifest.model_version.empty()) {
    throw std::invalid_argument("model name and version are required");
  }

  for (const auto& value : document.at("inputs")) {
    manifest.inputs.push_back(parse_tensor_spec(value));
  }
  for (const auto& value : document.at("outputs")) {
    manifest.outputs.push_back(parse_tensor_spec(value));
  }
  if (manifest.inputs.empty() || manifest.outputs.empty()) {
    throw std::invalid_argument("at least one input and output are required");
  }
  ensure_unique_names(manifest.inputs, "input");
  ensure_unique_names(manifest.outputs, "output");

  for (const auto& value : document.at("artifacts")) {
    ArtifactSpec artifact;
    artifact.backend = value.at("backend").get<std::string>();
    artifact.target =
        parse_device_type(value.at("target").get<std::string>());
    artifact.precision = value.at("precision").get<std::string>();
    artifact.runtime_version = value.value("runtime_version", "");
    artifact.path = value.at("path").get<std::string>();
    artifact.required_operators =
        parse_required_operators(value.value("required_operators", Json::array()));
    if (artifact.backend.empty() || artifact.precision.empty() ||
        !safe_relative_path(artifact.path)) {
      throw std::invalid_argument("invalid artifact entry");
    }
    const auto artifact_path =
        std::filesystem::canonical(canonical_root / artifact.path);
    if (!path_is_within(artifact_path, canonical_root) ||
        !std::filesystem::is_regular_file(artifact_path)) {
      throw std::invalid_argument("artifact does not exist: " +
                                  artifact.path.string());
    }
    manifest.artifacts.push_back(std::move(artifact));
  }
  if (manifest.artifacts.empty()) {
    throw std::invalid_argument("at least one artifact is required");
  }

  ModelPackage package;
  package.root_ = canonical_root;
  package.manifest_ = std::move(manifest);
  return package;
}

std::filesystem::path ModelPackage::artifact_path(
    const ArtifactSpec& artifact) const {
  return root_ / artifact.path;
}

}  // namespace physicsnemo::inference
