#include "transolver_workflow/runner.hpp"

#include <ATen/BlasBackend.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    // Give the runner a nontrivial starting pool and deliberately non-strict
    // math settings. CPU resource limits must not select a different math policy.
    at::set_num_threads(3);
    auto& context = at::globalContext();
    context.setFloat32MatmulPrecision("high");
    context.setBlasPreferredBackend(at::BlasBackend::Cublas);
    context.setAllowTF32CuBLAS(true);
    context.setAllowTF32CuDNN(true);
    context.setBenchmarkCuDNN(true);
    context.setDeterministicCuDNN(false);
    context.setDeterministicAlgorithms(false, false);
    transolver_workflow::RunnerConfig config;
    config.packages = {"missing-runtime-init-test-package"};
    config.math_profile = "provider-default-v1";
    // CPU-only: initialization precedes CUDA/package discovery. The deliberate
    // missing package also makes this safe to execute on a CUDA-capable host.
    for (int attempt = 0; attempt < 2; ++attempt) {
      bool expected_failure = false;
      std::string failure;
      try {
        transolver_workflow::Runner runner(config);
      } catch (const std::exception& error) {
        failure = error.what();
        expected_failure = failure.find("CUDA was requested but is unavailable") != std::string::npos ||
                           failure.find("missing-runtime-init-test-package") != std::string::npos;
      }
      check(expected_failure, "unexpected runner initialization result: " + failure);
      // The assertions below, rather than dependency absence, establish red.
      check(at::get_num_threads() == 1, "native CPU intra-op pool must be capped at one");
      check(at::get_num_interop_threads() == 1, "native CPU inter-op pool must be capped at one");
      for (const auto* name : {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "VTK_SMP_MAX_THREADS"}) {
        const auto* value = std::getenv(name);
        check(value && std::string(value) == "1", std::string(name) + " must cap native CPU resources");
      }
      check(context.allowTF32CuBLAS() && context.allowTF32CuDNN() &&
                context.benchmarkCuDNN() && !context.deterministicCuDNN() &&
                !context.deterministicAlgorithms() &&
                context.float32MatmulPrecision() == at::Float32MatmulPrecision::HIGH &&
                context.blasPreferredBackend() == at::BlasBackend::Cublas,
            "provider math settings changed while initializing CPU resources or loading a backend");
      // The AOTI SDK currently applies these overrides when loading CUDA
      // artifacts. Simulate them without CUDA; the process math policy must
      // survive a subsequent backend/Runner initialization, including failure.
      if (attempt == 0) {
        context.setFloat32MatmulPrecision("highest");
        context.setAllowTF32CuBLAS(false);
        context.setAllowTF32CuDNN(false);
      }
    }
    std::cout << "Native CPU initialization preserves provider math settings\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
