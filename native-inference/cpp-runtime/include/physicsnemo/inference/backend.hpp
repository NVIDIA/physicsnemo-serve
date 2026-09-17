#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

struct SessionCapabilities {
  bool accepts_cpu_inputs{true};
  bool accepts_device_inputs{false};
  bool caller_owned_cpu_outputs{true};
  bool caller_owned_device_outputs{false};
  bool backend_owned_device_outputs{false};
};

struct SessionOptions {
  std::optional<std::string> backend;
  Device device{};
  std::string precision{"auto"};
};

class BackendSession {
 public:
  virtual ~BackendSession() = default;
  virtual SessionCapabilities capabilities() const;
  virtual std::vector<OwnedTensor> run(
      const std::vector<TensorView>& inputs) = 0;

  // Retains backend-owned output storage without forcing it onto the CPU.
  // Device work is complete when this synchronous method returns.
  virtual std::vector<SharedTensor> run_owned(
      const std::vector<TensorView>& inputs);

  // Executes synchronously into caller-owned contiguous storage. Input and
  // output allocations must remain valid until this method returns. A session
  // is not safe for concurrent calls unless a backend documents otherwise.
  virtual void run_into(const std::vector<TensorView>& inputs,
                        const std::vector<MutableTensorView>& outputs);
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual bool supports(const ArtifactSpec& artifact,
                        const SessionOptions& options) const = 0;
  virtual std::unique_ptr<BackendSession> create_session(
      const ModelPackage& package, const ArtifactSpec& artifact,
      const SessionOptions& options) const = 0;
};

std::unique_ptr<Backend> create_mock_backend();

}  // namespace physicsnemo::inference
