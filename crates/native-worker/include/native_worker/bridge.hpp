#pragma once
#include "rust/cxx.h"
#ifndef PNMS_GENERIC_ONLY
#include "transolver_workflow/runner.hpp"
#endif
#include "physicsnemo/workflow/v1/host.hpp"
#include <memory>

namespace physicsnemo::serve {
struct RunnerConfig;
struct RunnerRequest;
struct Cancellation;
class CxxRunner {
 public:
  explicit CxxRunner(const RunnerConfig& config);
  explicit CxxRunner(const std::string& config_json);
  rust::String run(const RunnerRequest& request, const Cancellation& cancellation);
  rust::String self_test(const Cancellation& cancellation);
  rust::String run_workflow(rust::Str request, const Cancellation& cancellation);
 private:
#ifndef PNMS_GENERIC_ONLY
  std::unique_ptr<transolver_workflow::Runner> runner_;
#endif
  std::unique_ptr<physicsnemo::workflow::v1::WorkflowHost> workflow_;
};
std::unique_ptr<CxxRunner> create_runner(const RunnerConfig& config);
std::unique_ptr<CxxRunner> create_workflow(rust::Str config_json);
}
