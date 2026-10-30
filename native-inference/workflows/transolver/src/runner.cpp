// Persistent native workflow host. CLI behavior extracted from gpu_programming 87b78bf.
#include <ATen/BlasBackend.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/torch.h>
#include <torch/version.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
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
#include "physicsnemo/inference/v1/self_test.hpp"
#include "transolver_workflow/workflow.hpp"
#include "transolver_workflow/runner.hpp"
#include "transolver_workflow/request_plan.hpp"
#include "transolver_workflow/startup_check.hpp"

namespace {
using Json = nlohmann::json;
void configure_determinism() {
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
  stream.close();
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

void validate_output_paths(
    const std::filesystem::path& standardized_output,
    const std::filesystem::path& physical_output,
    const std::filesystem::path& metadata) {
  std::vector<std::pair<std::string, std::filesystem::path>> outputs{
      {"--standardized-output", standardized_output},
      {"--physical-output", physical_output},
      {"--metadata", metadata},
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
  stream.close();
  if (!stream) {
    throw std::runtime_error("failed to write metadata file: " + path.string());
  }
}


struct MathPolicy {
  bool tf32_matmul, tf32_cudnn, benchmark, cudnn_deterministic, deterministic;
  at::Float32MatmulPrecision precision;
  at::BlasBackend blas;

  static MathPolicy capture() {
    auto& context = at::globalContext();
    return {context.allowTF32CuBLAS(), context.allowTF32CuDNN(),
            context.benchmarkCuDNN(), context.deterministicCuDNN(),
            context.deterministicAlgorithms(), context.float32MatmulPrecision(),
            context.blasPreferredBackend()};
  }

  void apply() const {
    auto& context = at::globalContext();
    context.setBenchmarkCuDNN(benchmark);
    context.setDeterministicCuDNN(cudnn_deterministic);
    context.setDeterministicAlgorithms(deterministic, false);
    context.setBlasPreferredBackend(blas);
    context.setAllowTF32CuDNN(tf32_cudnn);
    context.setAllowTF32CuBLAS(tf32_matmul);
    // The TF32 compatibility setter also changes matmul precision, so restore
    // the recorded precision last (including a possible "medium" setting).
    context.setFloat32MatmulPrecision(
        precision == at::Float32MatmulPrecision::HIGH ? "high" :
        precision == at::Float32MatmulPrecision::MEDIUM ? "medium" : "highest");
  }
};

struct RestoreMathPolicy {
  MathPolicy policy;
  ~RestoreMathPolicy() { policy.apply(); }
};

MathPolicy initialize_process(const std::string& math_profile) {
  static std::mutex mutex;
  static std::string initialized_profile;
  static std::optional<MathPolicy> initialized_math;
  std::lock_guard lock(mutex);
  if (!initialized_profile.empty()) {
    if (initialized_profile != math_profile) {
      throw std::invalid_argument("one native process cannot mix math profiles");
    }
    return *initialized_math;
  }
  if (math_profile != "strict-native-v1" && math_profile != "provider-default-v1") {
    throw std::invalid_argument("unsupported native math profile: " + math_profile);
  }
  // Native process resource policy is independent of numerical policy. VTK/TBB
  // and ATen must not infer unbounded host pools from a container's CPU topology.
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
  setenv("VTK_SMP_MAX_THREADS", "1", 1);
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  if (math_profile == "strict-native-v1") {
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
    configure_determinism();
  }
  initialized_math = MathPolicy::capture();
  initialized_profile = math_profile;
  return *initialized_math;
}

void check_cancelled(const transolver_workflow::CancellationToken& token) {
  if (token.cancelled()) throw std::runtime_error("native workflow cancelled");
}

void write_i64(const std::filesystem::path& path, const torch::Tensor& tensor) {
  const auto contiguous = tensor.detach().to(torch::kCPU).contiguous();
  if (contiguous.scalar_type() != torch::kInt64) {
    throw std::invalid_argument("output tensor must be int64");
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create output file: " + path.string());
  stream.write(static_cast<const char*>(contiguous.const_data_ptr()),
               static_cast<std::streamsize>(contiguous.numel() * contiguous.element_size()));
  stream.close();
  if (!stream) throw std::runtime_error("failed to write output file: " + path.string());
}

Json math_flags() {
  auto& context = at::globalContext();
  std::string precision = "highest";
  switch (context.float32MatmulPrecision()) {
    case at::Float32MatmulPrecision::HIGH: precision = "high"; break;
    case at::Float32MatmulPrecision::MEDIUM: precision = "medium"; break;
    default: break;
  }
  std::string blas = "default";
  switch (context.blasPreferredBackend()) {
    case at::BlasBackend::Cublas: blas = "cublas"; break;
    case at::BlasBackend::Cublaslt: blas = "cublaslt"; break;
    case at::BlasBackend::Ck: blas = "ck"; break;
    default: break;
  }
  return {{"allow_tf32_matmul", context.allowTF32CuBLAS()},
          {"allow_tf32_cudnn", context.allowTF32CuDNN()},
          {"deterministic_algorithms", context.deterministicAlgorithms()},
          {"cudnn_deterministic", context.deterministicCuDNN()},
          {"cudnn_benchmark", context.benchmarkCuDNN()},
          {"float32_matmul_precision", precision},
          {"preferred_blas_library", blas}};
}

std::vector<float> read_startup_tensor(const Json& descriptor,
    const std::vector<std::int64_t>& shape, const std::filesystem::path& root) {
  if (std::endian::native != std::endian::little)
    throw std::invalid_argument("startup fixtures require a little-endian host");
  if (descriptor.at("shape") != Json(shape))
    throw std::invalid_argument("startup fixture shape differs from resident package");
  const auto relative = std::filesystem::path(descriptor.at("path").get<std::string>());
  if (relative.empty() || relative.is_absolute())
    throw std::invalid_argument("startup fixture tensor paths must be relative");
  auto path = root;
  for (const auto& part : relative) {
    if (part == ".." || part == ".")
      throw std::invalid_argument("startup fixture tensor path escapes its directory");
    path /= part;
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)))
      throw std::invalid_argument("startup fixture tensor path contains a symlink");
  }
  const auto count = physicsnemo::inference::tensor_byte_size(
      shape, physicsnemo::inference::DType::kFloat32) / sizeof(float);
  if (count > 16777216 || !std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != count * sizeof(float))
    throw std::invalid_argument("startup fixture tensor byte count differs or exceeds 64MiB");
  std::vector<float> values(count);
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(count*sizeof(float))))
    throw std::runtime_error("cannot read startup fixture tensor");
  transolver_workflow::validate_startup_output(shape,shape,values);
  return values;
}

}  // namespace

namespace transolver_workflow {

class Runner::Impl {
 public:
  explicit Impl(RunnerConfig value)
      : config(std::move(value)), domain(parse_domain(config.domain)),
        device(torch::kCUDA, config.device_index) {
    if (config.block_size < 2) throw std::invalid_argument("block_size must be at least two");
    if (config.device_index < 0) throw std::invalid_argument("device_index must not be negative");
    if (config.packages.empty()) throw std::invalid_argument("at least one model package is required");
    if (config.backend != "aoti" && config.backend != "tensorrt") {
      throw std::invalid_argument("backend must be aoti or tensorrt");
    }
    const auto load_start = std::chrono::steady_clock::now();
    // SDK loaders may select their own export/reference math settings. Keep
    // their loading behavior, then restore the workflow's configured process
    // policy on both successful loading and exceptions. The resident sessions
    // execute with this recorded policy, independently of export precision.
    const RestoreMathPolicy restore_math{initialize_process(config.math_profile)};
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA was requested but is unavailable");
    c10::cuda::CUDAGuard device_guard(device);
    c10::InferenceMode inference_mode;
    cudaDeviceProp properties{};
    const auto properties_status = cudaGetDeviceProperties(&properties, config.device_index);
    if (properties_status != cudaSuccess) {
      throw std::runtime_error("cannot identify CUDA deployment device: " +
                               std::string(cudaGetErrorString(properties_status)));
    }
    int cuda_version = 0;
    const auto runtime_status = cudaRuntimeGetVersion(&cuda_version);
    if (runtime_status != cudaSuccess) throw std::runtime_error("cannot identify CUDA runtime");
    device_metadata = {{"name", properties.name},
                       {"compute_capability", {properties.major, properties.minor}},
                       {"cuda_runtime_version", cuda_version},
                       {"torch_version", TORCH_VERSION}};
    for (const auto& path : config.packages) package_paths.emplace_back(path);
    const auto& backend = config.backend;
    const auto block_size = config.block_size;
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
    session_options.device = {physicsnemo::inference::DeviceType::kCuda, config.device_index};
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

    if (!config.stats_path.empty()) {
      stats = load_normalization_stats(config.stats_path, domain, device);
    }
    load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
  }

  std::string self_test(const CancellationToken& cancellation) {
    std::lock_guard lock(request_mutex);
    check_cancelled(cancellation);
    c10::cuda::CUDAGuard device_guard(device);
    c10::InferenceMode inference_mode;
    const auto start = std::chrono::steady_clock::now();
    Json fixtures = Json::object();
    std::filesystem::path fixture_root;
    if (!config.startup_fixture_path.empty()) {
      const auto path = std::filesystem::canonical(config.startup_fixture_path);
      fixture_root = path.parent_path();
      std::ifstream stream(path);
      const auto fixture = Json::parse(stream);
      std::vector<std::int64_t> package_points;
      for (const auto& selected : sessions)
        package_points.push_back(selected.point_dimension);
      const auto startup_points = startup_fixture_points(package_points, config.block_size);
      if (fixture.at("schema_version") != 1 || !fixture.at("models").is_array() ||
          fixture.at("models").size() != startup_points.size())
        throw std::invalid_argument("startup fixtures must cover every distinct resident point count exactly");
      for (const auto& model : fixture.at("models")) {
        const auto points = model.at("point_count").get<std::int64_t>();
        const auto key = std::to_string(points);
        if (points < 2 || points > config.block_size || fixtures.contains(key))
          throw std::invalid_argument("startup fixture point_count is invalid or duplicated");
        fixtures[key] = model;
      }
    }
    Json models = Json::array();
    for (auto& selected : sessions) {
      check_cancelled(cancellation);
      const auto points = selected.point_dimension == -1 ? config.block_size : selected.point_dimension;
      const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
      torch::Tensor fx, embedding;
      std::vector<float> expected;
      const Json* fixture = nullptr;
      if (config.startup_fixture_path.empty()) {
        fx = torch::full({1, points, 2}, 0.25, options);
        embedding = torch::full({1, points, embedding_width(domain)}, 0.125, options);
      } else {
        const auto key = std::to_string(points);
        if (!fixtures.contains(key)) throw std::invalid_argument("startup fixture missing resident package shape");
        fixture = &fixtures.at(key);
        auto fx_values = read_startup_tensor(fixture->at("inputs").at("fx"),{1,points,2},fixture_root);
        auto embedding_values = read_startup_tensor(fixture->at("inputs").at("embedding"),
            {1,points,embedding_width(domain)},fixture_root);
        const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        fx = torch::from_blob(fx_values.data(),{1,points,2},cpu_options).to(device);
        embedding = torch::from_blob(embedding_values.data(),{1,points,embedding_width(domain)},cpu_options).to(device);
        expected = read_startup_tensor(fixture->at("expected_output"),{1,points,output_width(domain)},fixture_root);
      }
      const std::vector<physicsnemo::inference::TensorView> inputs{
          tensor_view("fx", fx), tensor_view("embedding", embedding)};
      torch::Tensor output;
      std::vector<physicsnemo::inference::SharedTensor> owned;
      if (selected.session->capabilities().backend_owned_device_outputs) {
        owned = selected.session->run_owned(inputs);
        if (owned.size() != 1U) throw std::runtime_error("startup model returned wrong output count");
        output = output_tensor(owned.front().view());
      } else {
        output = torch::empty({1, points, output_width(domain)}, options);
        selected.session->run_into(inputs, {mutable_tensor_view(selected.output_name, output)});
      }
      if (output.scalar_type() != torch::kFloat32)
        throw std::runtime_error("startup model output must be float32");
      const auto cpu = output.to(torch::kCPU).contiguous();
      const std::vector<std::int64_t> shape{1, points, output_width(domain)};
      const auto actual = std::span<const float>(cpu.const_data_ptr<float>(), static_cast<std::size_t>(cpu.numel()));
      validate_startup_output(cpu.sizes().vec(), shape, actual);
      Json comparison = nullptr;
      if (fixture) {
        const auto rtol = fixture->at("rtol").get<double>();
        const auto atol = fixture->at("atol").get<double>();
        const auto compared = physicsnemo::inference::v1::compare_self_test_float32(actual,expected,rtol,atol);
        comparison = {{"rtol",rtol},{"atol",atol},{"max_absolute_error",compared.max_absolute_error},
                      {"max_relative_error",compared.max_relative_error}};
      }
      check_cancelled(cancellation);
      models.push_back({{"package", selected.path.string()}, {"backend", config.backend},
          {"point_count",points},{"shape", shape}, {"dtype", "float32"}, {"finite", true},
          {"comparison",comparison}});
    }
    return Json{{"status", "passed"}, {"policy", config.startup_fixture_path.empty() ? "finite-static-inputs-v1" : "expected-fp32-v1"},
        {"fixture",config.startup_fixture_path},
        {"session_load_count", sessions.size()}, {"models", models},
        {"device", device_metadata}, {"math_flags", math_flags()},
        {"duration_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count()}}.dump();
  }

  RunnerResult run(const RunnerRequest& request, const CancellationToken& cancellation) {
    // The worker dispatch gate is primary; this lock also protects direct embedders.
    std::lock_guard lock(request_mutex);
    check_cancelled(cancellation);
    if (request.mesh_path.empty() || request.stl_path.empty()) {
      throw std::invalid_argument("mesh_path and stl_path are required");
    }
    if (request.point_limit < 0 || request.permutation_seed < 0) {
      throw std::invalid_argument("point_limit and permutation_seed must not be negative");
    }
    if (!std::isfinite(request.air_density) || !std::isfinite(request.stream_velocity)) {
      throw std::invalid_argument("flow conditions must be finite");
    }
    const std::filesystem::path mesh_path(request.mesh_path), stl_path(request.stl_path);
    const std::filesystem::path standardized_output(request.standardized_output);
    const std::filesystem::path physical_output(request.physical_output);
    const std::filesystem::path metadata(request.metadata_path);
    if (standardized_output.empty() && physical_output.empty()) {
      throw std::invalid_argument("at least one standardized or physical output is required");
    }
    validate_output_paths(standardized_output, physical_output, metadata);
    if (!physical_output.empty() && !stats) {
      throw std::invalid_argument("physical output requires normalization statistics");
    }
    c10::cuda::CUDAGuard device_guard(device);
    c10::InferenceMode inference_mode;
    const auto point_limit = request.point_limit;
    const auto seed = request.permutation_seed;
    const auto block_size = config.block_size;
    const auto& backend = config.backend;
    const auto air_density = request.air_density;
    const auto stream_velocity = request.stream_velocity;
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
    const auto blocks = plan_blocks(point_count, block_size);
    check_cancelled(cancellation);
    const auto dump_directory = request.dump_input_dir;
    if (!dump_directory.empty()) {
      const std::filesystem::path directory(dump_directory);
      std::filesystem::create_directories(directory);
      write_f32(directory / "fx.f32", prepared.fx.contiguous());
      write_f32(
          directory / "embedding.f32", prepared.embedding.contiguous());
    }

    const auto preparation_end = std::chrono::steady_clock::now();
    std::vector<std::int64_t> package_points;
    for (const auto& session : sessions) package_points.push_back(session.point_dimension);
    preflight_blocks(blocks, package_points);
    auto generator = at::cuda::detail::createCUDAGenerator(config.device_index);
    generator.set_current_seed(static_cast<std::uint64_t>(seed));
    auto permutation = at::randperm(point_count, generator,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    if (!dump_directory.empty()) {
      write_i64(std::filesystem::path(dump_directory) / "permutation.i64", permutation);
      write_i64(std::filesystem::path(dump_directory) / "source_cell_ids.i64",
                torch::arange(point_count, torch::TensorOptions().dtype(torch::kInt64)));
    }
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
    for (const auto& block : blocks) {
      check_cancelled(cancellation);
      const auto begin = block.begin;
      const auto end = begin + block.points;
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
    check_cancelled(cancellation);

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
    Json payload = {
            {"schema_version", 2},
            {"domain", transolver_workflow::to_string(domain)},
            {"mesh", std::filesystem::absolute(mesh_path).string()},
            {"stl", std::filesystem::absolute(stl_path).string()},
            {"package", package_metadata_compatibility},
            {"packages", package_metadata},
            {"package_profiles", package_profiles},
            {"weights", ""},
            {"stats_path", config.stats_path.empty() ? "" : std::filesystem::absolute(config.stats_path).string()},
            {"dump_input_dir", dump_directory.empty() ? "" : std::filesystem::absolute(dump_directory).string()},
            {"backend", backend},
            {"device", "cuda:" + std::to_string(config.device_index)},
            {"math_profile", config.math_profile},
            {"deployment", device_metadata},
            {"math_flags", math_flags()},
            {"cpu_resources", {{"torch_intra_op_threads", at::get_num_threads()},
                               {"torch_inter_op_threads", at::get_num_interop_threads()},
                               {"omp_num_threads", "1"},
                               {"mkl_num_threads", "1"},
                               {"vtk_smp_max_threads", "1"}}},
            {"session_load_count", sessions.size()},
            {"initialization_ms", load_ms},
            {"timing_scope", "preparation excludes resident session/statistics load; inference includes gathers, decode and source-order CPU copies"},
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
        };
    payload["block_plan"] = Json::array();
    for (const auto& block : blocks) {
      payload["block_plan"].push_back({{"begin", block.begin}, {"points", block.points}});
    }
    if (!metadata.empty()) write_metadata(metadata, payload);
    return {payload.dump(), point_count, block_count, preparation_ms, inference_ms};

  }

 private:
  RunnerConfig config;
  Domain domain;
  torch::Device device;
  physicsnemo::inference::Runtime runtime;
  std::vector<std::filesystem::path> package_paths;
  std::vector<PackageSession> sessions;
  std::optional<NormalizationStats> stats;
  std::mutex request_mutex;
  double load_ms{0};
  Json device_metadata;
};

Runner::Runner(RunnerConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Runner::~Runner() = default;
std::string Runner::self_test(const CancellationToken& cancellation) {
  return impl_->self_test(cancellation);
}
RunnerResult Runner::run(const RunnerRequest& request, const CancellationToken& cancellation) {
  return impl_->run(request, cancellation);
}

}  // namespace transolver_workflow
