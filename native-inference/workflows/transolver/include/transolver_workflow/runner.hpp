#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace transolver_workflow {

// This boundary deliberately exposes no Torch, SDK or CUDA objects. Construct,
// execute and destroy a Runner on the worker's dedicated native thread.
struct RunnerConfig {
  std::vector<std::string> packages;
  std::string stats_path;
  std::string backend{"aoti"};
  std::string domain{"surface"};
  std::string math_profile{"provider-default-v1"};
  std::int64_t block_size{60000};
  std::int32_t device_index{0};
  // Authored raw-FP32 Builder/Python reference fixtures for every package shape.
  // Empty preserves the explicitly-labelled legacy finite-only startup smoke.
  std::string startup_fixture_path;
};

struct RunnerRequest {
  std::string mesh_path;
  std::string stl_path;
  std::string standardized_output;
  std::string physical_output;
  std::string metadata_path;
  std::string dump_input_dir;
  std::int64_t point_limit{0};
  // Effective CUDA seed. Derive the provider's per-case seed before calling run.
  std::int64_t permutation_seed{0};
  double air_density{1.205};
  double stream_velocity{30.0};
};

class CancellationToken {
 public:
  explicit CancellationToken(std::function<bool()> poll = {}) : poll_(std::move(poll)) {}
  void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
  bool cancelled() const {
    return cancelled_.load(std::memory_order_acquire) || (poll_ && poll_());
  }

 private:
  std::atomic<bool> cancelled_{false};
  std::function<bool()> poll_;
};

struct RunnerResult {
  std::string metadata_json;
  std::int64_t point_count{0};
  std::int64_t block_count{0};
  double preparation_ms{0};
  // Includes gathers, GPU execution, decoding and source-order CPU restoration.
  double inference_ms{0};
};

class Runner {
 public:
  explicit Runner(RunnerConfig config);
  ~Runner();
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;
  Runner(Runner&&) = delete;
  Runner& operator=(Runner&&) = delete;

  // Execute every resident package without a mesh, network or compilation.
  // startup_fixture_path selects authored expected-output numerical checks;
  // an empty path selects the explicitly labelled legacy finite-only smoke.
  // Neither policy replaces the full workflow scientific parity qualification.
  std::string self_test(const CancellationToken& cancellation);

  // Cooperative cancellation between stages/blocks cannot preempt a CUDA kernel.
  // No result files are published after an observed cancellation checkpoint.
  RunnerResult run(const RunnerRequest& request, const CancellationToken& cancellation);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace transolver_workflow
