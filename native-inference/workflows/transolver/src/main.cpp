// Migrated from gpu_programming commit 87b78bf, Transolver workflow CLI.
#include <ATen/BlasBackend.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/torch.h>

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

#include <nlohmann/json.hpp>
#include <vtkVersion.h>

#ifdef PNMIR_HAS_AOTI
#include "physicsnemo/inference/backends/aoti.hpp"
#endif
#ifdef PNMIR_HAS_TENSORRT
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/runtime.hpp"
#include "transolver_workflow/workflow.hpp"

namespace {

using Arguments =
    std::unordered_map<std::string, std::vector<std::string>>;
using Json = nlohmann::json;

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

void configure_determinism() {
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  auto& context = at::globalContext();
  context.setBenchmarkCuDNN(false);
  context.setDeterministicCuDNN(true);
  context.setDeterministicAlgorithms(true, false);
  context.setBlasPreferredBackend(at::BlasBackend::Cublas);
  context.setFloat32MatmulPrecision("highest");
  context.setAllowTF32CuBLAS(false);
  context.setAllowTF32CuDNN(false);
}

void write_f32(
    const std::filesystem::path& path,
    const torch::Tensor& tensor) {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  const auto contiguous =
      tensor.detach().to(torch::kCPU).contiguous();
  if (contiguous.scalar_type() != torch::kFloat32) {
    throw std::invalid_argument("output tensor must be float32");
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot create output file: " + path.string());
  }
  stream.write(
      static_cast<const char*>(contiguous.const_data_ptr()),
      static_cast<std::streamsize>(
          contiguous.numel() * contiguous.element_size()));
  if (!stream) {
    throw std::runtime_error("failed to write output file: " + path.string());
  }
}

physicsnemo::inference::TensorView tensor_view(
    const std::string& name,
    const torch::Tensor& tensor) {
  if (!tensor.is_cuda() || tensor.scalar_type() != torch::kFloat32 ||
      !tensor.is_contiguous()) {
    throw std::invalid_argument(
        name + " must be a contiguous CUDA float32 tensor");
  }
  physicsnemo::inference::Shape shape(tensor.sizes().begin(), tensor.sizes().end());
  return {
      name,
      physicsnemo::inference::DType::kFloat32,
      {physicsnemo::inference::DeviceType::kCuda, tensor.get_device()},
      std::move(shape),
      tensor.const_data_ptr(),
      static_cast<std::size_t>(tensor.numel() * tensor.element_size()),
  };
}

physicsnemo::inference::MutableTensorView mutable_tensor_view(
    const std::string& name,
    const torch::Tensor& tensor) {
  if (!tensor.is_cuda() || tensor.scalar_type() != torch::kFloat32 ||
      !tensor.is_contiguous()) {
    throw std::invalid_argument(
        name + " must be a contiguous CUDA float32 tensor");
  }
  physicsnemo::inference::Shape shape(tensor.sizes().begin(), tensor.sizes().end());
  return {
      name,
      physicsnemo::inference::DType::kFloat32,
      {physicsnemo::inference::DeviceType::kCuda, tensor.get_device()},
      std::move(shape),
      tensor.data_ptr(),
      static_cast<std::size_t>(tensor.numel() * tensor.element_size()),
  };
}

torch::Tensor output_tensor(const physicsnemo::inference::TensorView& view) {
  if (view.dtype != physicsnemo::inference::DType::kFloat32 ||
      view.device.type != physicsnemo::inference::DeviceType::kCuda) {
    throw std::runtime_error("PhysicsNeMo Inference returned a non-CUDA-float32 output");
  }
  const auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(torch::Device(torch::kCUDA, view.device.index));
  return torch::from_blob(
      const_cast<void*>(view.data), view.shape, options);
}

const physicsnemo::inference::TensorSpec& named_tensor(
    const std::vector<physicsnemo::inference::TensorSpec>& tensors,
    const std::string& name) {
  const auto found = std::find_if(
      tensors.begin(), tensors.end(), [&](const physicsnemo::inference::TensorSpec& tensor) {
        return tensor.name == name;
      });
  if (found == tensors.end()) {
    throw std::invalid_argument("package is missing tensor: " + name);
  }
  return *found;
}

std::int64_t package_point_dimension(const physicsnemo::inference::ModelManifest& manifest) {
  const auto fx_points = named_tensor(manifest.inputs, "fx").shape.at(1);
  const auto embedding_points =
      named_tensor(manifest.inputs, "embedding").shape.at(1);
  const auto output_points = manifest.outputs.at(0).shape.at(1);
  if (fx_points != embedding_points || fx_points != output_points) {
    throw std::invalid_argument(
        "package input and output point dimensions disagree");
  }
  return fx_points;
}

struct PackageSession {
  std::filesystem::path path;
  std::int64_t point_dimension{-1};
  std::string output_name;
  std::unique_ptr<physicsnemo::inference::InferenceSession> session;
};

PackageSession& select_session(
    std::vector<PackageSession>& sessions,
    const std::int64_t points) {
  const auto fixed = std::find_if(
      sessions.begin(), sessions.end(), [&](const PackageSession& candidate) {
        return candidate.point_dimension == points;
      });
  if (fixed != sessions.end()) {
    return *fixed;
  }
  const auto dynamic = std::find_if(
      sessions.begin(), sessions.end(), [](const PackageSession& candidate) {
        return candidate.point_dimension == -1;
      });
  if (dynamic != sessions.end()) {
    return *dynamic;
  }
  throw std::invalid_argument(
      "no package accepts a " + std::to_string(points) + "-point block");
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

void write_metadata(const std::filesystem::path& path, const Json& payload) {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot create metadata file: " + path.string());
  }
  stream << payload.dump(2) << '\n';
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
    const auto mesh_path =
        std::filesystem::path(require(arguments, "--mesh"));
    const auto stl_path = std::filesystem::path(require(arguments, "--stl"));
    const auto domain = transolver_workflow::parse_domain(
        require(arguments, "--domain"));
    const auto standardized_output =
        std::filesystem::path(optional(arguments, "--standardized-output"));
    const auto physical_output =
        std::filesystem::path(optional(arguments, "--physical-output"));
    if (standardized_output.empty() && physical_output.empty()) {
      throw std::invalid_argument(
          "at least one of --standardized-output or --physical-output is required");
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
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA was requested but is unavailable");
    }

    configure_determinism();
    c10::InferenceMode inference_mode;
    const torch::Device device(torch::kCUDA, 0);

    const auto preparation_start = std::chrono::steady_clock::now();
    auto prepared = transolver_workflow::prepare_inputs(
        mesh_path,
        stl_path,
        domain,
        device,
        static_cast<float>(air_density),
        static_cast<float>(stream_velocity),
        point_limit);
    const auto point_count = prepared.embedding.size(1);
    if (point_count < 2) {
      throw std::invalid_argument("Transolver requires at least two mesh locations");
    }
    if (point_count % block_size == 1) {
      throw std::invalid_argument(
          "the final block contains one point, outside the qualified dynamic package contract");
    }

    const auto dump_directory = optional(arguments, "--dump-input-dir");
    if (!dump_directory.empty()) {
      const std::filesystem::path directory(dump_directory);
      std::filesystem::create_directories(directory);
      write_f32(directory / "fx.f32", prepared.fx.contiguous());
      write_f32(
          directory / "embedding.f32", prepared.embedding.contiguous());
    }

    std::optional<transolver_workflow::NormalizationStats> stats;
    if (!physical_output.empty()) {
      stats = transolver_workflow::load_normalization_stats(
          require(arguments, "--stats"), domain, device);
    }
    const auto preparation_end = std::chrono::steady_clock::now();

    physicsnemo::inference::Runtime runtime;
    std::vector<PackageSession> sessions;
    if (backend == "aoti") {
#ifdef PNMIR_HAS_AOTI
      physicsnemo::inference::register_aoti_exact_operators(runtime);
      runtime.register_backend(physicsnemo::inference::create_aoti_backend());
#else
      throw std::runtime_error(
          "this workflow was built without AOTI support");
#endif
    } else {
#ifdef PNMIR_HAS_TENSORRT
      physicsnemo::inference::register_tensorrt_exact_operators(runtime);
      runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
#else
      throw std::runtime_error(
          "this workflow was built without TensorRT support");
#endif
    }
    physicsnemo::inference::SessionOptions session_options;
    session_options.backend = backend;
    session_options.device = {physicsnemo::inference::DeviceType::kCuda, 0};
    for (const auto& path : package_paths) {
      const auto package = physicsnemo::inference::ModelPackage::load(path);
      const auto package_points = package_point_dimension(package.manifest());
      const auto validation_points =
          package_points == -1 ? block_size : package_points;
      transolver_workflow::validate_package_contract(
          package.manifest(), domain, validation_points);
      if (package_points > block_size) {
        throw std::invalid_argument(
            "package point dimension exceeds --block-size: " +
            path.string());
      }
      if (std::ranges::any_of(
              sessions, [&](const PackageSession& existing) {
                return existing.point_dimension == package_points;
              })) {
        throw std::invalid_argument(
            "multiple packages have the same point dimension");
      }
      auto session = runtime.create_session(package, session_options);
      const auto capabilities = session->capabilities();
      if (!capabilities.accepts_device_inputs ||
          (!capabilities.backend_owned_device_outputs &&
           !capabilities.caller_owned_device_outputs)) {
        throw std::runtime_error(
            "selected PhysicsNeMo Inference backend does not support the device-resident workflow contract");
      }
      sessions.push_back(
          {path,
           package_points,
           package.manifest().outputs.front().name,
           std::move(session)});
    }
    for (std::int64_t begin = 0; begin < point_count; begin += block_size) {
      const auto points = std::min(block_size, point_count - begin);
      (void)select_session(sessions, points);
    }

    torch::manual_seed(seed);
    auto permutation = torch::randperm(
        point_count,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto restored_standardized = standardized_output.empty()
                                     ? torch::Tensor()
                                     : torch::empty(
                                           {point_count,
                                            transolver_workflow::output_width(domain)},
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(torch::kCPU));
    auto restored_physical = physical_output.empty()
                                 ? torch::Tensor()
                                 : torch::empty(
                                       {point_count,
                                        transolver_workflow::output_width(domain)},
                                       torch::TensorOptions()
                                           .dtype(torch::kFloat32)
                                           .device(torch::kCPU));

    const auto inference_start = std::chrono::steady_clock::now();
    std::int64_t block_count = 0;
    for (std::int64_t begin = 0; begin < point_count; begin += block_size) {
      const auto end = std::min(point_count, begin + block_size);
      auto indices = permutation.slice(0, begin, end);
      auto local_fx = prepared.fx.index_select(1, indices).contiguous();
      auto local_embedding =
          prepared.embedding.index_select(1, indices).contiguous();
      torch::Tensor standardized;
      std::vector<physicsnemo::inference::SharedTensor> outputs;
      const std::vector<physicsnemo::inference::TensorView> inputs{
          tensor_view("fx", local_fx),
          tensor_view("embedding", local_embedding),
      };
      auto& selected = select_session(sessions, end - begin);
      if (selected.session->capabilities().backend_owned_device_outputs) {
        outputs = selected.session->run_owned(inputs);
        if (outputs.size() != 1U) {
          throw std::runtime_error("PhysicsNeMo Inference returned the wrong output count");
        }
        standardized = output_tensor(outputs.front().view());
      } else {
        standardized = torch::empty(
            {1,
             end - begin,
             transolver_workflow::output_width(domain)},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(device));
        std::vector<physicsnemo::inference::MutableTensorView> output_views{
            mutable_tensor_view(selected.output_name, standardized),
        };
        selected.session->run_into(inputs, output_views);
      }
      const std::vector<std::int64_t> expected_shape{
          1, end - begin, transolver_workflow::output_width(domain)};
      if (standardized.sizes().vec() != expected_shape) {
        throw std::runtime_error("PhysicsNeMo Inference returned the wrong Transolver output shape");
      }
      auto cpu_indices = indices.to(torch::kCPU);
      if (restored_standardized.defined()) {
        restored_standardized.index_copy_(
            0, cpu_indices, standardized.squeeze(0).to(torch::kCPU));
      }
      if (restored_physical.defined()) {
        auto physical = transolver_workflow::decode_physical_outputs(
            standardized,
            *stats,
            domain,
            air_density,
            stream_velocity,
            prepared.volume_length_scale);
        restored_physical.index_copy_(
            0, cpu_indices, physical.squeeze(0).to(torch::kCPU));
      }
      ++block_count;
    }
    const auto inference_end = std::chrono::steady_clock::now();

    if (restored_standardized.defined()) {
      write_f32(standardized_output, restored_standardized);
    }
    if (restored_physical.defined()) {
      write_f32(physical_output, restored_physical);
    }

    const double preparation_ms =
        std::chrono::duration<double, std::milli>(
            preparation_end - preparation_start)
            .count();
    const double inference_ms =
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start)
            .count();
    const auto metadata = metadata_path(
        arguments, standardized_output, physical_output);
    Json package_metadata = Json::array();
    for (const auto& path : package_paths) {
      package_metadata.push_back(std::filesystem::absolute(path).string());
    }
    Json package_profiles = Json::array();
    for (const auto& session : sessions) {
      package_profiles.push_back(
          {{"path", std::filesystem::absolute(session.path).string()},
           {"point_dimension", session.point_dimension}});
    }
    const std::string package_metadata_compatibility =
        package_paths.size() == 1U
            ? std::filesystem::absolute(package_paths.front()).string()
            : "";
    write_metadata(
        metadata,
        {
            {"schema_version", 2},
            {"domain", transolver_workflow::to_string(domain)},
            {"mesh", std::filesystem::absolute(mesh_path).string()},
            {"stl", std::filesystem::absolute(stl_path).string()},
            {"package", package_metadata_compatibility},
            {"packages", package_metadata},
            {"package_profiles", package_profiles},
            {"weights", ""},
            {"backend", backend},
            {"device", "cuda:0"},
            {"mesh_reader", prepared.mesh_reader},
            {"point_count", point_count},
            {"point_limit", point_limit},
            {"block_size", block_size},
            {"block_count", block_count},
            {"permutation", "torch.cuda.randperm"},
            {"permutation_seed", seed},
            {"air_density", air_density},
            {"stream_velocity", stream_velocity},
            {"volume_length_scale", prepared.volume_length_scale},
            {"output_shape",
             {1, point_count, transolver_workflow::output_width(domain)}},
            {"output_dtype", "float32"},
            {"standardized_output",
             standardized_output.empty()
                 ? ""
                 : std::filesystem::absolute(standardized_output).string()},
            {"physical_output",
             physical_output.empty()
                 ? ""
                 : std::filesystem::absolute(physical_output).string()},
            {"vtk_version", vtkVersion::GetVTKVersion()},
            {"warp_version", PNMIR_WARP_VERSION},
            {"preparation_ms", preparation_ms},
            {"inference_ms", inference_ms},
        });
    std::cout << "physicsnemo-transolver domain="
              << transolver_workflow::to_string(domain)
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
