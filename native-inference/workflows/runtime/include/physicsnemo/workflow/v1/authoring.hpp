#pragma once

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "physicsnemo/inference/v1/api.hpp"

namespace physicsnemo::workflow::v1 {

struct Request {
  std::string request_id;
  std::string parameters_json;
};

struct Outcome {
  std::string payload_json{"{}"};
  std::string metadata_json{"{}"};
};

// Identity checked by Rust admission. The path comes only from admitted inputs;
// the metadata accessor does not download data or substitute a caller path.
struct AdmittedInput {
  std::string artifact_id;
  std::filesystem::path path;
  std::uint64_t size_bytes{0};
  std::string sha256;
  std::string media_type;
  std::string source;
};

// This context is borrowed only for initialize(). Parse configuration/assets
// into owned workflow members; do not retain the context itself.
class InitializationContext {
 public:
  const std::string& configuration_json() const;
  const inference::ModelManifest& model(const std::string& alias) const;
  std::filesystem::path asset(const std::string& configured_name) const;
  void check_cancelled() const;

 private:
  class Impl;
  explicit InitializationContext(Impl* impl) : impl_(impl) {}
  Impl* impl_;
  friend class WorkflowHost;
};

// Request-scoped context. Borrowed inputs and the context cannot escape run().
class Context {
 public:
  const std::string& configuration_json() const;
  // Blocking inference returns owned CPU output storage, valid as long as the
  // returned Result is retained. Aliases are fixed by deployment configuration.
  inference::v1::Result infer(const std::string& alias,
                            const std::vector<inference::TensorView>& inputs);
  const inference::ModelManifest& model(const std::string& alias) const;
  void check_cancelled() const;
  std::filesystem::path input(const std::string& admitted_name) const;
  AdmittedInput input_info(const std::string& admitted_name) const;
  std::filesystem::path asset(const std::string& configured_name) const;
  std::filesystem::path output_path(const std::string& relative_path) const;
  // Names use at most 128 ASCII letters/digits or '_', '-', '.' (not '.'/'..').
  // Media types use at most 128 bytes without CR/LF. Rust rechecks this boundary.
  void register_output(const std::string& relative_path, const std::string& name,
                       const std::string& media_type, bool primary = false);

 private:
  class Impl;
  explicit Context(Impl* impl) : impl_(impl) {}
  Impl* impl_;
  friend class WorkflowHost;
};

class Workflow {
 public:
  virtual ~Workflow() = default;
  // Called once after model checks, on the host's owning thread. Exceptions or
  // cancellation fail startup permanently. Destruction uses that same thread.
  virtual void initialize(InitializationContext&) {}
  virtual Outcome run(const Request& request, Context& context) = 0;
};

// A custom source-linked target supplies exactly this symbol. The host selects
// it only when built with PNMS_CUSTOM_WORKFLOW_SOURCE_DIR/TARGET and configured
// with adapter="custom.v1". No runtime dynamic plugin loader is involved.
std::unique_ptr<Workflow> create_custom_workflow();

}  // namespace physicsnemo::workflow::v1
