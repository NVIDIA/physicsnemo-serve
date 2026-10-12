#include "physicsnemo/inference/backend.hpp"

#include <stdexcept>
#include <utility>

namespace physicsnemo::inference {
namespace {

class MockSession final : public BackendSession {
 public:
  explicit MockSession(std::vector<TensorSpec> outputs)
      : outputs_(std::move(outputs)) {}

  std::vector<OwnedTensor> run(
      const std::vector<TensorView>& inputs) override {
    if (inputs.size() != 1 || outputs_.size() != 1) {
      throw std::runtime_error("mock backend supports one input and one output");
    }
    const auto& input = inputs.front();
    const auto& output = outputs_.front();
    if (input.dtype != output.dtype || input.shape.size() != output.shape.size()) {
      throw std::runtime_error("mock identity input/output schemas differ");
    }

    std::vector<std::byte> storage(input.byte_size);
    const auto* source = static_cast<const std::byte*>(input.data);
    std::copy(source, source + input.byte_size, storage.begin());
    return {OwnedTensor(output.name, output.dtype, input.device, input.shape,
                        std::move(storage))};
  }

 private:
  std::vector<TensorSpec> outputs_;
};

class MockBackend final : public Backend {
 public:
  std::string name() const override { return "mock"; }

  bool supports(const ArtifactSpec& artifact,
                const SessionOptions& options) const override {
    const bool precision_matches =
        options.precision == "auto" || options.precision == artifact.precision;
    return artifact.backend == name() && artifact.target == options.device.type &&
           options.device.type == DeviceType::kCpu && precision_matches;
  }

  std::unique_ptr<BackendSession> create_session(
      const ModelPackage& package, const ArtifactSpec&,
      const SessionOptions&) const override {
    return std::make_unique<MockSession>(package.manifest().outputs);
  }
};

}  // namespace

std::unique_ptr<Backend> create_mock_backend() {
  return std::make_unique<MockBackend>();
}

}  // namespace physicsnemo::inference
