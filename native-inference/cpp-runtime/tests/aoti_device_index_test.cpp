#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <iostream>
#include <stdexcept>

#include "physicsnemo/inference/backends/aoti.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace {

namespace pnm = physicsnemo::inference;

void check_current_device(int expected) {
  if (c10::cuda::current_device() != expected) {
    throw std::runtime_error("AOTI changed the caller's current CUDA device");
  }
}

void check_inference(const pnm::Runtime& runtime,
                     const pnm::ModelPackage& package, int device_index,
                     int current_device) {
  pnm::SessionOptions options;
  options.backend = "aoti";
  options.device = {pnm::DeviceType::kCuda, device_index};
  auto session = runtime.create_session(package, options);
  check_current_device(current_device);

  auto input = torch::tensor({1.0F, 2.0F, 3.0F});
  const pnm::TensorView view{"input", pnm::DType::kFloat32, {}, {3},
                             input.const_data_ptr(), 3 * sizeof(float)};
  const auto outputs = session->run_owned({view});
  check_current_device(current_device);
  if (outputs.size() != 1) {
    throw std::runtime_error("AOTI returned an incorrect output count");
  }
  const auto output = outputs.front().view();
  if (output.device != options.device || output.dtype != pnm::DType::kFloat32 ||
      output.shape != pnm::Shape{3}) {
    throw std::runtime_error("AOTI output is not on the requested CUDA device");
  }
  const auto actual = torch::from_blob(
      const_cast<void*>(output.data), output.shape,
      torch::TensorOptions().dtype(torch::kFloat32).device(
          torch::Device(torch::kCUDA, device_index))).cpu();
  if (!torch::equal(actual, torch::tensor({3.0F, 5.0F, 13.0F}))) {
    throw std::runtime_error("AOTI weighted inference returned incorrect values");
  }
  session.reset();
  check_current_device(current_device);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: pnmir_aoti_device_index_test PACKAGE");
    }
    const int device_count = c10::cuda::device_count();
    if (device_count == 0) {
      std::cout << "CUDA is unavailable; skipping AOTI device index test\n";
      return 77;
    }
    const int current_device = c10::cuda::current_device();
    const auto package = pnm::ModelPackage::load(argv[1]);
    pnm::Runtime runtime;
    runtime.register_backend(pnm::create_aoti_backend());
    check_inference(runtime, package, current_device, current_device);
    if (device_count > 1) {
      const auto caller_stream =
          c10::cuda::getStreamFromPool(false, current_device);
      const c10::cuda::CUDAStreamGuard stream_guard(caller_stream);
      check_inference(runtime, package, (current_device + 1) % device_count,
                      current_device);
      if (c10::cuda::getCurrentCUDAStream(current_device) != caller_stream) {
        throw std::runtime_error("AOTI changed the caller's current CUDA stream");
      }
    }

    // A weighted package must allocate its constants on the requested device
    // during loading, so even a single-GPU host detects an ignored index.
    pnm::SessionOptions invalid;
    invalid.backend = "aoti";
    bool ignored_index = false;
    // -1 is Torch's current-device sentinel, and 256 wraps to zero when
    // converted to its signed 8-bit DeviceIndex without validation.
    for (const int index : {device_count, -1, 256}) {
      invalid.device = {pnm::DeviceType::kCuda, index};
      bool rejected = false;
      try {
        runtime.create_session(package, invalid);
      } catch (const std::exception& error) {
        rejected = true;
        std::cout << "AOTI rejected CUDA device " << index
                  << " during session creation: " << error.what() << '\n';
      }
      if (!rejected) {
        std::cerr << "AOTI accepted invalid CUDA device index " << index << '\n';
        ignored_index = true;
      }
      check_current_device(current_device);
    }
    if (ignored_index) {
      throw std::runtime_error(
          "AOTI session creation ignored an invalid CUDA device index");
    }
    std::cout << "AOTI explicit CUDA device selection passed on " << device_count
              << " device(s)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
