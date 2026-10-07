#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "device_runner_count.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/runtime.hpp"

#ifdef PNMIR_HAS_AOTI
#include "physicsnemo/inference/backends/aoti.hpp"
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
#include "physicsnemo/inference/backends/onnxruntime.hpp"
#endif
#ifdef PNMIR_HAS_TENSORRT
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif

namespace {

void check_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " +
                             cudaGetErrorString(status));
  }
}

std::filesystem::path tensor_file_path(const std::filesystem::path& directory,
                                       const std::string& name) {
  if (name.empty() || std::filesystem::path(name).has_root_path() ||
      name.find_first_of("/\\") != std::string::npos ||
      name.find('\0') != std::string::npos) {
    throw std::invalid_argument("tensor name must be a filename: " + name);
  }
  return directory / (name + ".bin");
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path,
                                  std::size_t expected_size) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open tensor file: " + path.string());
  }
  const auto end = input.tellg();
  if (end < 0 || static_cast<std::size_t>(end) != expected_size) {
    throw std::runtime_error("tensor file size mismatch: " + path.string());
  }
  std::vector<std::byte> result(expected_size);
  input.seekg(0);
  if (expected_size != 0 &&
      !input.read(reinterpret_cast<char*>(result.data()),
                  static_cast<std::streamsize>(result.size()))) {
    throw std::runtime_error("cannot read tensor file: " + path.string());
  }
  return result;
}

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::byte>& storage) {
  std::ofstream output(path, std::ios::binary);
  if (!output ||
      (storage.size() != 0 &&
       !output.write(reinterpret_cast<const char*>(storage.data()),
                     static_cast<std::streamsize>(storage.size())))) {
    throw std::runtime_error("cannot write tensor file: " + path.string());
  }
  output.close();
  if (!output) {
    throw std::runtime_error("cannot write tensor file: " + path.string());
  }
}

std::shared_ptr<void> allocate_cuda(std::size_t byte_size) {
  void* pointer = nullptr;
  check_cuda(cudaMalloc(&pointer, byte_size), "cudaMalloc");
  return {pointer, [](void* value) {
            if (value != nullptr) static_cast<void>(cudaFree(value));
          }};
}

physicsnemo::inference::Shape parse_shape(const std::string& text) {
  physicsnemo::inference::Shape shape;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      throw std::invalid_argument("shape dimensions cannot be empty");
    }
    std::size_t parsed = 0;
    const auto dimension = std::stoll(token, &parsed);
    if (parsed != token.size() || dimension <= 0) {
      throw std::invalid_argument("shape dimensions must be positive");
    }
    shape.push_back(dimension);
  }
  if (shape.empty()) throw std::invalid_argument("shape cannot be empty");
  return shape;
}

std::pair<std::string, physicsnemo::inference::Shape> parse_named_shape(
    const std::string& text) {
  const auto separator = text.find('=');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == text.size()) {
    throw std::invalid_argument(
        "--shape must use the form NAME=d0,d1,...");
  }
  return {text.substr(0, separator), parse_shape(text.substr(separator + 1))};
}

physicsnemo::inference::Shape resolve_shape(
    const physicsnemo::inference::TensorSpec& spec,
    const std::unordered_map<std::string, physicsnemo::inference::Shape>& overrides) {
  const auto found = overrides.find(spec.name);
  if (found == overrides.end()) {
    if (std::ranges::find(spec.shape, -1) != spec.shape.end()) {
      throw std::invalid_argument(
          "dynamic tensor requires --shape: " + spec.name);
    }
    return spec.shape;
  }
  const auto& shape = found->second;
  if (shape.size() != spec.shape.size()) {
    throw std::invalid_argument("shape rank mismatch for tensor: " + spec.name);
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (spec.shape[i] != -1 && spec.shape[i] != shape[i]) {
      throw std::invalid_argument(
          "shape mismatch for tensor: " + spec.name);
    }
  }
  return shape;
}

std::size_t tensor_bytes(physicsnemo::inference::DType dtype,
                         const physicsnemo::inference::Shape& shape) {
  return physicsnemo::inference::tensor_byte_size(shape, dtype);
}

double percentile(std::vector<double> values, std::size_t numerator) {
  std::ranges::sort(values);
  const std::size_t rank =
      std::max<std::size_t>(1, (numerator * values.size() + 99) / 100);
  return values[rank - 1];
}

struct OutputStorage {
  physicsnemo::inference::TensorSpec spec;
  physicsnemo::inference::Device device;
  std::shared_ptr<void> owner;
  void* data;
  std::size_t byte_size;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 7) {
      throw std::invalid_argument(
          "usage: physicsnemo-infer-device-runner PACKAGE BACKEND INPUT_DIR OUTPUT_DIR "
          "WARMUP ITERATIONS [--shape NAME=d0,d1,...]...");
    }
    const std::filesystem::path package_path(argv[1]);
    const std::string backend(argv[2]);
    const std::filesystem::path input_dir(argv[3]);
    const std::filesystem::path output_dir(argv[4]);
    const std::size_t warmup =
        device_runner::parse_count(argv[5], "warmup", true);
    const std::size_t iterations =
        device_runner::parse_count(argv[6], "iterations", false);
    std::unordered_map<std::string, physicsnemo::inference::Shape> shape_overrides;
    for (int i = 7; i < argc; i += 2) {
      if (std::string(argv[i]) != "--shape" || i + 1 >= argc) {
        throw std::invalid_argument(
            "expected repeated --shape NAME=d0,d1,... arguments");
      }
      auto [name, shape] = parse_named_shape(argv[i + 1]);
      if (!shape_overrides.emplace(std::move(name), std::move(shape)).second) {
        throw std::invalid_argument("duplicate --shape tensor name");
      }
    }

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    const auto package = physicsnemo::inference::ModelPackage::load(package_path);
    physicsnemo::inference::Runtime runtime;
#ifdef PNMIR_HAS_AOTI
    physicsnemo::inference::register_aoti_exact_operators(runtime);
    runtime.register_backend(physicsnemo::inference::create_aoti_backend());
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
    runtime.register_backend(physicsnemo::inference::create_onnxruntime_backend());
#endif
#ifdef PNMIR_HAS_TENSORRT
    physicsnemo::inference::register_tensorrt_exact_operators(runtime);
    runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
#endif
    physicsnemo::inference::SessionOptions options;
    options.backend = backend;
    options.device = {physicsnemo::inference::DeviceType::kCuda, 0};
    auto session = runtime.create_session(package, options);
    const auto capabilities = session->capabilities();
    if (!capabilities.accepts_device_inputs) {
      throw std::runtime_error(
          "selected backend does not accept device-resident inputs");
    }
    const bool use_backend_owned_outputs =
        !capabilities.caller_owned_device_outputs &&
        capabilities.backend_owned_device_outputs;

    std::vector<physicsnemo::inference::SharedTensor> prepared_tensors;
    prepared_tensors.reserve(package.manifest().inputs.size());
    for (const auto& spec : package.manifest().inputs) {
      const auto shape = resolve_shape(spec, shape_overrides);
      const std::size_t byte_size = tensor_bytes(spec.dtype, shape);
      const auto host =
          read_bytes(tensor_file_path(input_dir, spec.name), byte_size);
      auto owner = allocate_cuda(byte_size);
      check_cuda(cudaMemcpy(owner.get(), host.data(), byte_size,
                            cudaMemcpyHostToDevice),
                 "prepared input upload");
      prepared_tensors.emplace_back(
          spec.name, spec.dtype,
          physicsnemo::inference::Device{physicsnemo::inference::DeviceType::kCuda, 0}, shape,
          owner.get(), byte_size, std::move(owner));
    }
    const physicsnemo::inference::PreparedState prepared(package.manifest().model_name,
                                          std::move(prepared_tensors));
    const auto inputs = prepared.views();

    std::vector<OutputStorage> output_storage;
    output_storage.reserve(package.manifest().outputs.size());
    std::vector<physicsnemo::inference::MutableTensorView> outputs;
    outputs.reserve(package.manifest().outputs.size());
    if (!use_backend_owned_outputs) {
      for (const auto& spec : package.manifest().outputs) {
        auto resolved_spec = spec;
        resolved_spec.shape = resolve_shape(spec, shape_overrides);
        const std::size_t byte_size =
            tensor_bytes(resolved_spec.dtype, resolved_spec.shape);
        if (capabilities.caller_owned_device_outputs) {
          auto owner = allocate_cuda(byte_size);
          void* data = owner.get();
          output_storage.push_back(
              {resolved_spec, {physicsnemo::inference::DeviceType::kCuda, 0},
               std::move(owner), data, byte_size});
        } else {
          auto storage = std::make_shared<std::vector<std::byte>>(byte_size);
          void* data = storage->data();
          std::shared_ptr<void> owner = storage;
          output_storage.push_back(
              {resolved_spec, physicsnemo::inference::Device{}, std::move(owner), data,
               byte_size});
        }
        const auto& storage = output_storage.back();
        outputs.push_back(
            {storage.spec.name, storage.spec.dtype, storage.device,
             storage.spec.shape, storage.data, storage.byte_size});
      }
    }

    std::vector<physicsnemo::inference::SharedTensor> retained_outputs;
    const auto run_once = [&]() {
      if (use_backend_owned_outputs) {
        retained_outputs = session->run_owned(inputs);
        for (const auto& output : retained_outputs) {
          const auto view = output.view();
          if (view.device !=
              physicsnemo::inference::Device{physicsnemo::inference::DeviceType::kCuda, 0}) {
            throw std::runtime_error(
                "backend-owned output is not on the session CUDA device");
          }
        }
      } else {
        session->run_into(inputs, outputs);
      }
      check_cuda(cudaDeviceSynchronize(), "inference synchronization");
    };

    for (std::size_t i = 0; i < warmup; ++i) {
      run_once();
    }
    std::vector<double> durations_ms;
    durations_ms.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
      const auto start = std::chrono::steady_clock::now();
      run_once();
      const auto end = std::chrono::steady_clock::now();
      durations_ms.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::filesystem::create_directories(output_dir);
    const auto write_output = [&](const physicsnemo::inference::TensorView& output) {
      std::vector<std::byte> host(output.byte_size);
      if (output.device.type == physicsnemo::inference::DeviceType::kCuda) {
        check_cuda(cudaMemcpy(host.data(), output.data, output.byte_size,
                              cudaMemcpyDeviceToHost),
                   "final output download");
      } else {
        std::copy_n(static_cast<const std::byte*>(output.data),
                    output.byte_size, host.begin());
      }
      write_bytes(tensor_file_path(output_dir, output.name), host);
    };
    if (use_backend_owned_outputs) {
      for (const auto& output : retained_outputs) {
        write_output(output.view());
      }
    } else {
      for (const auto& output : output_storage) {
        write_output({output.spec.name, output.spec.dtype, output.device,
                      output.spec.shape, output.data, output.byte_size});
      }
    }

    const double mean =
        std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0) /
        static_cast<double>(durations_ms.size());
    std::cout << std::fixed << std::setprecision(3)
              << "benchmark: warmup=" << warmup
              << " iterations=" << iterations
              << " prepared_bytes=" << prepared.byte_size()
              << " device_outputs="
              << (use_backend_owned_outputs ||
                          capabilities.caller_owned_device_outputs
                      ? 1
                      : 0)
              << " mean_ms=" << mean
              << " p50_ms=" << percentile(durations_ms, 50)
              << " p95_ms=" << percentile(durations_ms, 95) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
