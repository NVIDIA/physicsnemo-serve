#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef PNMIR_HAS_AOTI
#include "physicsnemo/inference/backends/aoti.hpp"
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
#include "physicsnemo/inference/backends/onnxruntime.hpp"
#endif
#ifdef PNMIR_HAS_TENSORRT
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace {

constexpr std::size_t kElementCount = 3;
constexpr std::size_t kByteSize = kElementCount * sizeof(float);
constexpr std::size_t kRunCount = 5;

void check_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t byte_size) : byte_size_(byte_size) {
    check_cuda(cudaMalloc(&data_, byte_size_), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) static_cast<void>(cudaFree(data_));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }
  std::size_t byte_size() const { return byte_size_; }

 private:
  void* data_{nullptr};
  std::size_t byte_size_;
};

void register_backend(physicsnemo::inference::Runtime& runtime, const std::string& backend) {
#ifdef PNMIR_HAS_AOTI
  if (backend == "aoti") {
    runtime.register_backend(physicsnemo::inference::create_aoti_backend());
    return;
  }
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
  if (backend == "onnxruntime") {
    runtime.register_backend(physicsnemo::inference::create_onnxruntime_backend());
    return;
  }
#endif
#ifdef PNMIR_HAS_TENSORRT
  if (backend == "tensorrt") {
    runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
    return;
  }
#endif
  throw std::invalid_argument(
      "persistent-state test backend is not compiled: " + backend);
}

void check_values(const std::vector<float>& actual) {
  const std::vector<float> expected{3.0F, 5.0F, 7.0F};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > 1.0e-6F) {
      throw std::runtime_error(
          "persistent-state output value mismatch");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      std::cout << "CUDA is unavailable; skipping persistent-state test\n";
      return 77;
    }
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: pnmir_persistent_device_state_test BACKEND PACKAGE");
    }

    const std::string backend(argv[1]);
    const auto package = physicsnemo::inference::ModelPackage::load(argv[2]);
    if (package.manifest().inputs.size() != 1 ||
        package.manifest().outputs.size() != 1) {
      throw std::invalid_argument(
          "persistent-state test requires one input and one output");
    }
    const auto& input_spec = package.manifest().inputs.front();
    const auto& output_spec = package.manifest().outputs.front();
    if (input_spec.name != "input" ||
        input_spec.dtype != physicsnemo::inference::DType::kFloat32 ||
        input_spec.shape != physicsnemo::inference::Shape{kElementCount} ||
        output_spec.name != "output" ||
        output_spec.dtype != physicsnemo::inference::DType::kFloat32 ||
        output_spec.shape != physicsnemo::inference::Shape{kElementCount}) {
      throw std::invalid_argument(
          "persistent-state test requires the affine [3] tensor contract");
    }

    physicsnemo::inference::Runtime runtime;
    register_backend(runtime, backend);
    physicsnemo::inference::SessionOptions options;
    options.backend = backend;
    options.device = {physicsnemo::inference::DeviceType::kCuda, 0};
    auto session = runtime.create_session(package, options);
    const auto capabilities = session->capabilities();
    if (!capabilities.accepts_device_inputs) {
      throw std::runtime_error(
          backend + " did not report caller-owned CUDA input support");
    }

    const std::vector<float> values{1.0F, 2.0F, 3.0F};
    auto input_buffer = std::make_shared<DeviceBuffer>(kByteSize);
    check_cuda(cudaMemcpy(input_buffer->data(), values.data(), kByteSize,
                          cudaMemcpyHostToDevice),
               "one-time input upload");
    const void* const persistent_input_address = input_buffer->data();
    const std::weak_ptr<DeviceBuffer> input_lifetime = input_buffer;
    std::shared_ptr<void> input_owner = input_buffer;

    {
      const physicsnemo::inference::PreparedState prepared(
          "affine-input-v1",
          {physicsnemo::inference::SharedTensor(
              input_spec.name, input_spec.dtype,
              {physicsnemo::inference::DeviceType::kCuda, 0}, input_spec.shape,
              persistent_input_address, kByteSize, std::move(input_owner))});
      input_buffer.reset();
      if (input_lifetime.expired()) {
        throw std::runtime_error(
            "PreparedState did not retain the CUDA input allocation");
      }
      const auto inputs = prepared.views();

      std::unique_ptr<DeviceBuffer> device_output;
      const bool use_backend_owned_outputs =
          !capabilities.caller_owned_device_outputs &&
          capabilities.backend_owned_device_outputs;
      if (capabilities.caller_owned_device_outputs) {
        device_output = std::make_unique<DeviceBuffer>(kByteSize);
      } else if (!use_backend_owned_outputs) {
        throw std::runtime_error(
            backend + " did not report a CUDA output path");
      }
      void* const persistent_output_address =
          device_output == nullptr ? nullptr : device_output->data();
      const physicsnemo::inference::MutableTensorView output{
          output_spec.name, output_spec.dtype,
          {physicsnemo::inference::DeviceType::kCuda, 0},
          output_spec.shape, persistent_output_address, kByteSize};

      for (std::size_t run = 0; run < kRunCount; ++run) {
        if (inputs.front().data != persistent_input_address) {
          throw std::runtime_error(
              "persistent input address changed between requests");
        }

        std::vector<float> actual(kElementCount);
        if (use_backend_owned_outputs) {
          const auto retained_outputs = session->run_owned(inputs);
          if (retained_outputs.size() != 1) {
            throw std::runtime_error(
                "backend-owned CUDA output count mismatch");
          }
          const auto retained = retained_outputs.front().view();
          if (retained.device !=
                  physicsnemo::inference::Device{physicsnemo::inference::DeviceType::kCuda, 0} ||
              retained.byte_size != kByteSize) {
            throw std::runtime_error(
                "backend-owned output is not the expected CUDA tensor");
          }
          check_cuda(cudaMemcpy(actual.data(), retained.data, kByteSize,
                                cudaMemcpyDeviceToHost),
                     "backend-owned output validation download");
        } else {
          if (output.data != persistent_output_address) {
            throw std::runtime_error(
                "persistent output address changed between requests");
          }
          check_cuda(cudaMemset(output.data, 0, kByteSize),
                     "clear CUDA output");
          session->run_into(inputs, {output});
          check_cuda(cudaMemcpy(actual.data(), output.data, kByteSize,
                                cudaMemcpyDeviceToHost),
                     "output validation download");
        }
        check_values(actual);
      }
    }

    if (!input_lifetime.expired()) {
      throw std::runtime_error(
          "PreparedState did not release the CUDA input allocation");
    }

    std::cout << backend << " persistent PreparedState CUDA reuse passed ("
              << kRunCount << " calls)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
