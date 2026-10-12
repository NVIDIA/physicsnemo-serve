#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "physicsnemo/inference/backends/tensorrt.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace {

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

std::vector<std::byte> as_bytes(const std::vector<float>& values) {
  std::vector<std::byte> result(values.size() * sizeof(float));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

void check_values(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("TensorRT output size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > 1.0e-6F) {
      throw std::runtime_error("TensorRT device output value mismatch");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      std::cout << "CUDA is unavailable; skipping device I/O test\n";
      return 77;
    }
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: pnmir_tensorrt_device_io_test PACKAGE");
    }

    const auto package = physicsnemo::inference::ModelPackage::load(argv[1]);
    physicsnemo::inference::Runtime runtime;
    runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
    physicsnemo::inference::SessionOptions options;
    options.backend = "tensorrt";
    options.device = {physicsnemo::inference::DeviceType::kCuda, 0};
    auto session = runtime.create_session(package, options);

    const auto capabilities = session->capabilities();
    if (!capabilities.accepts_device_inputs ||
        !capabilities.caller_owned_device_outputs) {
      throw std::runtime_error(
          "TensorRT did not report caller-owned CUDA I/O support");
    }

    constexpr std::size_t kElementCount = 3;
    constexpr std::size_t kByteSize = kElementCount * sizeof(float);
    auto input_buffer = std::make_shared<DeviceBuffer>(kByteSize);
    DeviceBuffer output_buffer(kByteSize);
    const physicsnemo::inference::Device device{physicsnemo::inference::DeviceType::kCuda, 0};
    std::shared_ptr<void> input_owner = input_buffer;
    const physicsnemo::inference::PreparedState prepared(
        "affine-input-v1",
        {physicsnemo::inference::SharedTensor("input", physicsnemo::inference::DType::kFloat32, device, {3},
                               input_buffer->data(),
                               input_buffer->byte_size(),
                               std::move(input_owner))});
    const physicsnemo::inference::TensorView input = prepared.views().front();
    const physicsnemo::inference::MutableTensorView output{
        "output", physicsnemo::inference::DType::kFloat32, device, {3},
        output_buffer.data(), output_buffer.byte_size()};

    for (const auto& values :
         {std::vector<float>{1.0F, 2.0F, 3.0F},
          std::vector<float>{-2.0F, 0.5F, 9.0F}}) {
      check_cuda(cudaMemcpy(input_buffer->data(), values.data(), kByteSize,
                            cudaMemcpyHostToDevice),
                 "input upload");
      session->run_into({input}, {output});
      std::vector<float> actual(kElementCount);
      check_cuda(cudaMemcpy(actual.data(), output_buffer.data(), kByteSize,
                            cudaMemcpyDeviceToHost),
                 "output download");
      std::vector<float> expected;
      expected.reserve(values.size());
      for (const float value : values) expected.push_back(2.0F * value + 1.0F);
      check_values(actual, expected);
    }

    auto wrong_device_input = input;
    wrong_device_input.device.index = 1;
    bool rejected = false;
    try {
      session->run_into({wrong_device_input}, {output});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error(
          "TensorRT accepted an input from a different CUDA device");
    }

    const std::vector<float> host_values{1.0F, 2.0F, 3.0F};
    physicsnemo::inference::OwnedTensor host_input(
        "input", physicsnemo::inference::DType::kFloat32, {}, {3}, as_bytes(host_values));
    const auto host_outputs = session->run({host_input.view()});
    std::vector<float> host_actual(kElementCount);
    std::memcpy(host_actual.data(), host_outputs[0].view().data, kByteSize);
    check_values(host_actual, {3.0F, 5.0F, 7.0F});

    std::cout << "TensorRT caller-owned CUDA I/O passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
