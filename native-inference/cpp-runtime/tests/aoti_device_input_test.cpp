#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "physicsnemo/inference/backends/aoti.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/runtime.hpp"

int main(int argc, char** argv) {
  try {
    if (!torch::cuda::is_available()) {
      std::cout << "CUDA is unavailable; skipping AOTI device input test\n";
      return 77;
    }
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: pnmir_aoti_device_input_test PACKAGE");
    }

    const auto package = physicsnemo::inference::ModelPackage::load(argv[1]);
    physicsnemo::inference::Runtime runtime;
    runtime.register_backend(physicsnemo::inference::create_aoti_backend());
    physicsnemo::inference::SessionOptions options;
    options.backend = "aoti";
    options.device = {physicsnemo::inference::DeviceType::kCuda, 0};
    auto session = runtime.create_session(package, options);

    const auto capabilities = session->capabilities();
    if (!capabilities.accepts_device_inputs ||
        capabilities.caller_owned_device_outputs ||
        !capabilities.backend_owned_device_outputs) {
      throw std::runtime_error(
          "AOTI CUDA session reported incorrect device I/O capabilities");
    }

    auto input_tensor =
        torch::tensor({1.0F, 2.0F, 3.0F},
                      torch::TensorOptions().device(torch::kCUDA));
    const physicsnemo::inference::TensorView input{
        "input",
        physicsnemo::inference::DType::kFloat32,
        {physicsnemo::inference::DeviceType::kCuda, 0},
        {3},
        input_tensor.const_data_ptr(),
        3 * sizeof(float),
    };
    std::vector<float> output_storage(3);
    const physicsnemo::inference::MutableTensorView output{
        "output",
        physicsnemo::inference::DType::kFloat32,
        {},
        {3},
        output_storage.data(),
        output_storage.size() * sizeof(float),
    };
    session->run_into({input}, {output});

    const std::vector<float> expected{3.0F, 5.0F, 7.0F};
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (std::abs(output_storage[i] - expected[i]) > 1.0e-6F) {
        throw std::runtime_error("AOTI device input output mismatch");
      }
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
          "AOTI accepted an input from a different CUDA device");
    }

    std::cout << "AOTI caller-owned CUDA input passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
