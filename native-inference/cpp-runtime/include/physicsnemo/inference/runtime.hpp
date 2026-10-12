#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

class InferenceSession {
 public:
  InferenceSession(ModelManifest manifest,
                   std::unique_ptr<BackendSession> backend_session,
                   std::string backend_name);

  SessionCapabilities capabilities() const;
  std::vector<OwnedTensor> run(const std::vector<TensorView>& inputs);
  std::vector<SharedTensor> run_owned(
      const std::vector<TensorView>& inputs);
  void run_into(const std::vector<TensorView>& inputs,
                const std::vector<MutableTensorView>& outputs);
  const std::string& backend_name() const { return backend_name_; }

 private:
  ModelManifest manifest_;
  std::unique_ptr<BackendSession> backend_session_;
  std::string backend_name_;
};

class Runtime {
 public:
  void register_backend(std::unique_ptr<Backend> backend);
  void register_operator(std::string id, std::string abi);
  std::unique_ptr<InferenceSession> create_session(
      const ModelPackage& package, const SessionOptions& options = {}) const;

 private:
  std::vector<std::unique_ptr<Backend>> backends_;
  std::unordered_map<std::string, std::string> operators_;
};

}  // namespace physicsnemo::inference
