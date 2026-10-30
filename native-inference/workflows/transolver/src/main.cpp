// Migrated from gpu_programming commit 87b78bf, Transolver workflow CLI.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transolver_workflow/runner.hpp"

namespace {

using Arguments =
    std::unordered_map<std::string, std::vector<std::string>>;

void usage() {
  std::cout
      << "PhysicsNeMo C++ Transolver workflow\n\n"
      << "usage: physicsnemo-transolver --package PATH [--package PATH ...]\n"
      << "       --mesh PATH --stl PATH --domain surface|volume\n"
      << "       --physical-output FILE --stats FILE [options]\n\n"
      << "Runs raw mesh preprocessing, native model inference, and physical-unit decoding.\n"
      << "Outputs are contiguous float32 values in source mesh order.\n\n"
      << "options:\n"
      << "  --backend aoti|tensorrt   Package backend (default: aoti)\n"
      << "  --package PATH           Backend package; repeat for fixed-size tail blocks\n"
      << "  --block-size N           Points per inference block (default: 2048)\n"
      << "  --point-limit N          Read the first N points; 0 reads all (default: 0)\n"
      << "  --seed N                 CUDA permutation seed (default: 0)\n"
      << "  --air-density VALUE      Air density (default: 1.205)\n"
      << "  --stream-velocity VALUE  Stream velocity (default: 30.0)\n"
      << "  --device cuda            Execution device (default: cuda)\n"
      << "  --math-profile PROFILE   strict-native-v1 (default) or provider-default-v1\n"
      << "  --startup-fixture FILE   Independent expected-output startup goldens\n"
      << "  --startup-report FILE    Write the resident-model startup report\n"
      << "  --startup-only true      Check resident models without mesh execution\n"
      << "  --stats FILE             Normalization statistics for physical output\n"
      << "  --physical-output FILE   Fields decoded to physical units\n"
      << "  --standardized-output FILE  Optional model-space fields\n"
      << "  --metadata FILE          JSON record (default: output filename + .json)\n"
      << "  --dump-input-dir DIR     Prepared tensor files for parity diagnostics\n"
      << "  --help                   Show this help\n\n"
      << "At least one output is required. Every block must contain at least two points.\n";
}

Arguments parse_arguments(const int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string option(argv[index]);
    if (!option.starts_with("--")) {
      throw std::invalid_argument("unexpected positional argument: " + option);
    }
    result[option].emplace_back(argv[index + 1]);
  }
  return result;
}

const std::string& require(
    const Arguments& arguments,
    const std::string& name) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    throw std::invalid_argument("missing required option: " + name);
  }
  if (found->second.size() != 1U) {
    throw std::invalid_argument("option may be specified only once: " + name);
  }
  return found->second.front();
}

std::string optional(
    const Arguments& arguments,
    const std::string& name,
    const std::string& default_value = {}) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    return default_value;
  }
  if (found->second.size() != 1U) {
    throw std::invalid_argument("option may be specified only once: " + name);
  }
  return found->second.front();
}

std::vector<std::string> repeated(
    const Arguments& arguments,
    const std::string& name) {
  const auto found = arguments.find(name);
  return found == arguments.end() ? std::vector<std::string>{}
                                  : found->second;
}

std::int64_t parse_integer(
    const std::string& value,
    const std::string& name,
    const bool allow_zero) {
  std::size_t parsed = 0;
  const auto result = std::stoll(value, &parsed);
  if (parsed != value.size() || result < 0 || (!allow_zero && result == 0)) {
    throw std::invalid_argument(name + " must be a valid positive integer");
  }
  return result;
}

double parse_number(const std::string& value, const std::string& name) {
  std::size_t parsed = 0;
  const double result = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument(name + " must be a finite number");
  }
  return result;
}

std::filesystem::path metadata_path(
    const Arguments& arguments,
    const std::filesystem::path& standardized_output,
    const std::filesystem::path& physical_output) {
  const auto explicit_path = optional(arguments, "--metadata");
  if (!explicit_path.empty()) {
    return explicit_path;
  }
  auto base = !physical_output.empty() ? physical_output : standardized_output;
  base += ".json";
  return base;
}

void validate_output_paths(
    const std::filesystem::path& standardized_output,
    const std::filesystem::path& physical_output,
    const std::filesystem::path& metadata,
    const std::filesystem::path& startup_report) {
  std::vector<std::pair<std::string, std::filesystem::path>> outputs{
      {"--standardized-output", standardized_output},
      {"--physical-output", physical_output},
      {"--metadata", metadata},
      {"--startup-report", startup_report},
  };
  for (auto& [option, path] : outputs) {
    if (!path.empty()) {
      std::filesystem::path resolved;
      for (const auto& component : std::filesystem::absolute(path)) {
        resolved /= component;
        if (std::filesystem::is_symlink(resolved) &&
            !std::filesystem::exists(resolved)) {
          throw std::invalid_argument(
              "output path contains a dangling symlink: " + option);
        }
        resolved = std::filesystem::weakly_canonical(resolved);
      }
      path = std::move(resolved);
    }
  }
  for (std::size_t first = 0; first < outputs.size(); ++first) {
    if (outputs[first].second.empty()) {
      continue;
    }
    for (std::size_t second = first + 1; second < outputs.size(); ++second) {
      if (outputs[second].second.empty()) {
        continue;
      }
      std::error_code error;
      if (outputs[first].second == outputs[second].second ||
          std::filesystem::equivalent(
              outputs[first].second, outputs[second].second, error)) {
        throw std::invalid_argument(
            "output paths must be distinct: " + outputs[first].first +
            " and " + outputs[second].first);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      usage();
      return 0;
    }
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
    setenv("OMP_NUM_THREADS", "1", 0);
    setenv("MKL_NUM_THREADS", "1", 0);
    setenv("VTK_SMP_MAX_THREADS", "1", 0);
    const auto arguments = parse_arguments(argc, argv);
    const std::string backend = optional(arguments, "--backend", "aoti");
    const auto package_values = repeated(arguments, "--package");
    std::vector<std::filesystem::path> package_paths;
    package_paths.reserve(package_values.size());
    for (const auto& value : package_values) {
      package_paths.emplace_back(value);
    }
    if (backend != "aoti" && backend != "tensorrt") {
      throw std::invalid_argument(
          "--backend must be aoti or tensorrt");
    }
    if (package_paths.empty()) {
      throw std::invalid_argument(
          "--backend " + backend + " requires at least one --package");
    }
    if (arguments.contains("--weights")) {
      throw std::invalid_argument(
          "this workflow consumes model packages; --weights is unsupported");
    }
    const auto startup_only = optional(arguments, "--startup-only", "false");
    if (startup_only != "true" && startup_only != "false") {
      throw std::invalid_argument("--startup-only must be true or false");
    }
    const bool check_only = startup_only == "true";
    const auto mesh_path = std::filesystem::path(
        check_only ? "" : require(arguments, "--mesh"));
    const auto stl_path = std::filesystem::path(
        check_only ? "" : require(arguments, "--stl"));
    const auto domain = require(arguments, "--domain");
    if (domain != "surface" && domain != "volume") {
      throw std::invalid_argument("--domain must be surface or volume");
    }
    const auto standardized_output =
        std::filesystem::path(optional(arguments, "--standardized-output"));
    const auto physical_output =
        std::filesystem::path(optional(arguments, "--physical-output"));
    if (!check_only && standardized_output.empty() && physical_output.empty()) {
      throw std::invalid_argument(
          "at least one of --standardized-output or --physical-output is required");
    }
    const auto metadata = check_only ? std::filesystem::path{} : metadata_path(
        arguments, standardized_output, physical_output);
    const auto startup_report = optional(arguments, "--startup-report");
    validate_output_paths(
        check_only ? std::filesystem::path{} : standardized_output,
        check_only ? std::filesystem::path{} : physical_output,
        metadata, startup_report);
    if (!startup_report.empty() && std::filesystem::exists(startup_report)) {
      throw std::invalid_argument("--startup-report already exists");
    }
    const auto block_size = parse_integer(
        optional(arguments, "--block-size", "2048"), "--block-size", false);
    const auto point_limit = parse_integer(
        optional(arguments, "--point-limit", "0"), "--point-limit", true);
    const auto seed = parse_integer(
        optional(arguments, "--seed", "0"), "--seed", true);
    const double air_density = parse_number(
        optional(arguments, "--air-density", "1.205"), "--air-density");
    const double stream_velocity = parse_number(
        optional(arguments, "--stream-velocity", "30.0"),
        "--stream-velocity");
    const std::string device_name = optional(arguments, "--device", "cuda");
    if (device_name != "cuda") {
      throw std::invalid_argument("this workflow requires --device cuda");
    }
    transolver_workflow::RunnerConfig config;
    for (const auto& path : package_paths) config.packages.push_back(path.string());
    config.stats_path = check_only ? optional(arguments, "--stats") :
        (physical_output.empty() ? "" : require(arguments, "--stats"));
    config.backend = backend;
    config.domain = domain;
    config.block_size = block_size;
    config.math_profile = optional(arguments, "--math-profile", "strict-native-v1");
    if (config.math_profile != "strict-native-v1" &&
        config.math_profile != "provider-default-v1") {
      throw std::invalid_argument("unsupported --math-profile");
    }
    config.startup_fixture_path = optional(arguments, "--startup-fixture");
    if (check_only && startup_report.empty()) {
      throw std::invalid_argument("--startup-only requires --startup-report");
    }
    const bool check_startup =
        !config.startup_fixture_path.empty() || !startup_report.empty();
    transolver_workflow::Runner runner(std::move(config));
    if (check_startup) {
      const auto report = runner.self_test(transolver_workflow::CancellationToken{});
      if (!startup_report.empty()) {
        std::ofstream stream(startup_report);
        stream << report << '\n';
        stream.close();
        if (!stream) throw std::runtime_error("cannot write --startup-report");
      }
    }
    if (check_only) return 0;
    transolver_workflow::RunnerRequest request;
    request.mesh_path = mesh_path.string();
    request.stl_path = stl_path.string();
    request.standardized_output = standardized_output.string();
    request.physical_output = physical_output.string();
    request.metadata_path = metadata.string();
    request.dump_input_dir = optional(arguments, "--dump-input-dir");
    request.point_limit = point_limit;
    request.permutation_seed = seed;
    request.air_density = air_density;
    request.stream_velocity = stream_velocity;
    const auto result = runner.run(request, transolver_workflow::CancellationToken{});
    const auto point_count = result.point_count;
    const auto block_count = result.block_count;
    const auto preparation_ms = result.preparation_ms;
    const auto inference_ms = result.inference_ms;
    std::cout << "physicsnemo-transolver domain="
              << domain
              << " points=" << point_count << " blocks=" << block_count
              << " preparation_ms=" << preparation_ms
              << " inference_ms=" << inference_ms
              << " metadata=" << metadata << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "physicsnemo-transolver: " << error.what() << '\n';
    return 1;
  }
}
