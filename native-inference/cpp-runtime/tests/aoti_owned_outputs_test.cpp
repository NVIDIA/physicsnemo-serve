#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "physicsnemo/inference/backends/aoti.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace {

namespace pnm = physicsnemo::inference;

void check_outputs(const std::vector<pnm::SharedTensor>& outputs,
                   pnm::Device target, const std::string& phase) {
  const std::vector<std::vector<float>> expected{
      {0, 1, 2, 3}, {1, 2}, {0, 2}, {11, 12}, {21, 22}};
  if (outputs.size() != expected.size()) {
    throw std::runtime_error("incorrect owned output count");
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto view = outputs[i].view();
    if (view.device != target || view.dtype != pnm::DType::kFloat32 ||
        view.shape != pnm::Shape{static_cast<std::int64_t>(expected[i].size())}) {
      throw std::runtime_error("incorrect owned output metadata: " + view.name);
    }
    const auto device = target.type == pnm::DeviceType::kCuda
                            ? torch::Device(torch::kCUDA, target.index)
                            : torch::Device(torch::kCPU);
    auto actual = torch::from_blob(
        const_cast<void*>(view.data), view.shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(device)).cpu();
    if (!torch::equal(actual, torch::tensor(expected[i]))) {
      throw std::runtime_error("owned output " + view.name + " changed " + phase);
    }
  }
}

void check_ownership(const pnm::ModelPackage& package, pnm::Device target,
                     torch::Device input_device) {
  std::vector<pnm::SharedTensor> outputs;
  {
    pnm::Runtime runtime;
    runtime.register_backend(pnm::create_aoti_backend());
    pnm::SessionOptions options;
    options.backend = "aoti";
    options.device = target;
    auto session = runtime.create_session(package, options);
    auto input = torch::arange(4, torch::TensorOptions().dtype(torch::kFloat32)
                                     .device(input_device));
    const pnm::TensorView view{
        "input", pnm::DType::kFloat32,
        input.is_cuda() ? target : pnm::Device{}, {4}, input.const_data_ptr(),
        4 * sizeof(float)};
    outputs = session->run_owned({view});
    check_outputs(outputs, target, "before input mutation");
    input.fill_(-99);
    check_outputs(outputs, target, "after caller input mutation");
    // A later invocation must not invalidate storage retained by earlier results.
    session->run_owned({view});
    check_outputs(outputs, target, "after session reuse");
  }
  check_outputs(outputs, target, "after input and session destruction");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: pnmir_aoti_owned_outputs_test PACKAGE cpu|cuda");
    }
    const auto package = pnm::ModelPackage::load(argv[1]);
    const bool cuda = std::string(argv[2]) == "cuda";
    const pnm::Device target{cuda ? pnm::DeviceType::kCuda : pnm::DeviceType::kCpu, 0};
    check_ownership(package, target, torch::Device(torch::kCPU));
    if (cuda) {
      check_ownership(package, target, torch::Device(torch::kCUDA, 0));
    }
    std::cout << "AOTI owned outputs retain values across input and session lifetimes\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
