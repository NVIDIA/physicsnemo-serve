#include "native-worker/src/bridge.rs.h"
#include "native_worker/bridge.hpp"
#include <stdexcept>

namespace physicsnemo::serve {
CxxRunner::CxxRunner(const std::string& config_json) {
  workflow_ = std::make_unique<physicsnemo::workflow::v1::WorkflowHost>(config_json);
}
rust::String CxxRunner::self_test(const Cancellation& cancellation) {
  if (workflow_) return rust::String(workflow_->self_test([&cancellation] { return cancellation.is_cancelled(); }));
#ifndef PNMS_GENERIC_ONLY
  transolver_workflow::CancellationToken token([&cancellation] { return cancellation.is_cancelled(); });
  return rust::String(runner_->self_test(token));
#else
  throw std::runtime_error("native workflow host is not initialized");
#endif
}
rust::String CxxRunner::run_workflow(rust::Str request, const Cancellation& cancellation) {
  return rust::String(workflow_->run(std::string(request), [&cancellation] { return cancellation.is_cancelled(); }));
}
CxxRunner::CxxRunner(const RunnerConfig& config) {
#ifndef PNMS_GENERIC_ONLY
  transolver_workflow::RunnerConfig native;
  for (const auto& package : config.packages) native.packages.emplace_back(std::string(package));
  native.stats_path = std::string(config.stats_path);
  native.startup_fixture_path = std::string(config.startup_fixture_path);
  native.backend = std::string(config.backend);
  native.domain = std::string(config.domain);
  native.math_profile = std::string(config.math_profile);
  native.block_size = config.block_size;
  native.device_index = config.device_index;
  runner_ = std::make_unique<transolver_workflow::Runner>(std::move(native));
#else
  (void)config;
  throw std::runtime_error("Transolver adapter is not linked into this generic workflow worker");
#endif
}
rust::String CxxRunner::run(const RunnerRequest& request, const Cancellation& cancellation) {
#ifndef PNMS_GENERIC_ONLY
  transolver_workflow::RunnerRequest native;
  native.mesh_path = std::string(request.mesh_path);
  native.stl_path = std::string(request.stl_path);
  native.standardized_output = std::string(request.standardized_output);
  native.physical_output = std::string(request.physical_output);
  native.metadata_path = std::string(request.metadata_path);
  native.dump_input_dir = std::string(request.dump_input_dir);
  native.permutation_seed = request.permutation_seed;
  native.air_density = request.air_density;
  native.stream_velocity = request.stream_velocity;
  transolver_workflow::CancellationToken token([&cancellation] { return cancellation.is_cancelled(); });
  return rust::String(runner_->run(native, token).metadata_json);
#else
  (void)request;
  (void)cancellation;
  throw std::runtime_error("Transolver adapter is not linked into this generic workflow worker");
#endif
}
std::unique_ptr<CxxRunner> create_runner(const RunnerConfig& config) {
  return std::make_unique<CxxRunner>(config);
}
std::unique_ptr<CxxRunner> create_workflow(rust::Str config_json) {
  return std::make_unique<CxxRunner>(std::string(config_json));
}
}
