#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/v1/api.hpp"
#ifdef PNMIR_QA_WITH_CUDA
#include "physicsnemo/inference/backends/aoti.hpp"
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif

namespace pnm = physicsnemo::inference;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    if (argc < 6) {
      throw std::invalid_argument(
          "usage: physicsnemo-qa-consumer PACKAGE BACKEND OUTPUT_DIR INPUT_FILE INPUT_FILE ...");
    }
    const std::string backend = argv[2];
    const fs::path output_dir = argv[3];
    const bool cpu = backend == "mock";
    pnm::v1::Engine engine;
    if (cpu) {
      engine.register_backend(pnm::create_mock_backend());
    }
#ifdef PNMIR_QA_WITH_CUDA
    else if (backend == "aoti") {
      engine.register_backend(pnm::create_aoti_backend());
    } else if (backend == "tensorrt") {
      engine.register_backend(pnm::create_tensorrt_backend());
    }
#endif
    else {
      throw std::invalid_argument("unsupported QA backend");
    }
    const auto model = engine.load_model(argv[1]);
    const auto& manifest = model.manifest();
    if (manifest.inputs.size() != 1 || manifest.outputs.size() != 1 ||
        manifest.inputs[0].name != "input" || manifest.outputs[0].name != "output" ||
        manifest.inputs[0].dtype != pnm::DType::kFloat32 ||
        manifest.outputs[0].dtype != pnm::DType::kFloat32 ||
        manifest.inputs[0].shape != pnm::Shape{4} ||
        manifest.outputs[0].shape != pnm::Shape{4}) {
      throw std::invalid_argument("QA consumer expects the configured-affine tensor contract");
    }
    pnm::v1::ExecutorOptions options;
    options.backend = backend;
    options.device = {cpu ? pnm::DeviceType::kCpu : pnm::DeviceType::kCuda, 0};
    auto executor = model.create_executor(options);
    if (executor.backend_name() != backend) {
      throw std::runtime_error("SDK selected a different backend");
    }
    if (fs::exists(output_dir / "metadata.json")) {
      throw std::invalid_argument("QA output must not contain prior completion metadata");
    }
    std::vector<pnm::v1::Result> retained;
    std::vector<std::vector<float>> snapshots;
    for (int argument = 4; argument < argc; ++argument) {
      const fs::path input_path = argv[argument];
      constexpr std::size_t bytes = 4 * sizeof(float);
      if (fs::file_size(input_path) != bytes) {
        throw std::invalid_argument("input file size does not match the affine contract");
      }
      auto input = std::make_shared<std::vector<float>>(4);
      std::ifstream stream(input_path, std::ios::binary);
      if (!stream.read(reinterpret_cast<char*>(input->data()), bytes)) {
        throw std::runtime_error("cannot read input");
      }
      auto output = std::make_shared<std::vector<float>>(4);
      pnm::v1::Request request;
      request.bind_input({"input", pnm::DType::kFloat32, {}, {4}, input->data(), bytes}, input);
      request.bind_output({"output", pnm::DType::kFloat32, {}, {4}, output->data(), bytes}, output);
      auto result = executor.run(request);
      const auto view = result.output("output");
      if (result.output_count() != 1 || view.dtype != pnm::DType::kFloat32 ||
          view.shape != pnm::Shape{4} || view.device != pnm::Device{} ||
          view.byte_size != bytes) {
        throw std::runtime_error("SDK output metadata mismatch");
      }
      std::vector<float> snapshot(4);
      std::memcpy(snapshot.data(), view.data, bytes);
      for (float value : snapshot) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite SDK output");
      }
      const auto output_path = output_dir / ("output-" + std::to_string(argument - 4) + ".f32");
      std::ofstream destination(output_path, std::ios::binary);
      if (!destination.write(reinterpret_cast<const char*>(view.data), bytes)) {
        throw std::runtime_error("cannot write SDK output");
      }
      retained.push_back(std::move(result));
      snapshots.push_back(std::move(snapshot));
    }
    // Caller buffers and the executor may be released; retained results must
    // preserve their values after subsequent requests and executor destruction.
    executor = {};
    for (std::size_t i = 0; i < retained.size(); ++i) {
      const auto view = retained[i].output("output");
      if (std::memcmp(view.data, snapshots[i].data(), view.byte_size) != 0) {
        throw std::runtime_error("earlier SDK result changed after executor reuse");
      }
    }
    std::ofstream metadata(output_dir / "metadata.json");
    metadata << "{\"completed\":true,\"backend\":\"" << backend
             << "\",\"execution_device\":{\"type\":\"" << (cpu ? "cpu" : "cuda")
             << "\",\"index\":0},\"executor_count\":1,\"requests\":" << argc - 4 << "}\n";
    if (!metadata) throw std::runtime_error("cannot write SDK completion metadata");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
