#include "transolver_workflow/runner.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F&& f, const std::string& fragment) {
  try { f(); } catch (const std::exception& error) {
    check(std::string(error.what()).find(fragment) != std::string::npos,
          "wrong error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection: " + fragment);
}
Json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read test config");
  return Json::parse(stream);
}
std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), {}};
}
}

int main(int argc, char** argv) {
  using namespace transolver_workflow;
  try {
    CancellationToken token;
    check(!token.cancelled(), "fresh token cancelled");
    token.cancel();
    check(token.cancelled(), "cancel must persist");
    bool externally_cancelled = false;
    CancellationToken callback([&] { return externally_cancelled; });
    check(!callback.cancelled(), "fresh callback cancelled");
    externally_cancelled = true;
    check(callback.cancelled(), "external cancellation was not polled");
    RunnerConfig invalid;
    invalid.block_size = 1;
    rejects([&] { Runner runner(invalid); }, "block_size must be at least two");
    invalid.block_size = 60000;
    rejects([&] { Runner runner(invalid); }, "at least one model package");
    // Explicit GPU fixture mode: this is separate from CPU contract CTest.
    // The fixture owns copied packages/stats and writable output_dir.
    if (argc == 1) return 0;
    if (argc != 2) throw std::invalid_argument("usage: transolver_runner_tests [fixture.json]");
    const auto fixture = read_json(argv[1]);
    RunnerConfig config;
    config.packages = fixture.at("packages").get<std::vector<std::string>>();
    config.stats_path = fixture.at("stats_path");
    config.block_size = fixture.at("block_size");
    config.math_profile = fixture.value("math_profile", "strict-native-v1");
    config.backend = fixture.value("backend", "aoti");
    config.domain = fixture.value("domain", "surface");
    config.device_index = fixture.value("device_index", 0);
    config.startup_fixture_path = fixture.value("startup_fixture_path", "");
    Runner runner(config);
    const auto startup = Json::parse(runner.self_test(CancellationToken{}));
    check(startup.at("status") == "passed" && startup.at("models").size() == config.packages.size(),
          "startup did not execute every resident model package");
    check(startup.at("session_load_count") == config.packages.size(),
          "startup reloaded package sessions");
    const std::filesystem::path root = fixture.at("output_dir").get<std::string>();
    RunnerRequest request;
    request.mesh_path = fixture.at("mesh_path");
    request.stl_path = fixture.at("stl_path");
    request.point_limit = fixture.at("point_limit");
    request.permutation_seed = fixture.value("permutation_seed", 0LL);
    request.air_density = fixture.value("air_density", 1.205);
    request.stream_velocity = fixture.value("stream_velocity", 30.0);
    request.dump_input_dir = fixture.value("dump_input_dir", (root / "inputs").string());
    request.metadata_path = fixture.value("metadata_path", (root / "metadata.json").string());
    request.standardized_output = (root / "first-standardized.f32").string();
    request.physical_output = (root / "first-physical.f32").string();
    const auto first = runner.run(request, CancellationToken{});
    const auto expected = read_bytes(request.physical_output);
    check(!expected.empty(), "first request produced no physical fields");
    // Only use owned copies: hide manifests and statistics after construction.
    // A second load or per-request stats read must fail instead of going unnoticed.
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> hidden;
    const auto hide = [&](const std::filesystem::path& path) {
      auto target = path;
      target += ".resident-test";
      std::filesystem::rename(path, target);
      hidden.emplace_back(path, target);
    };
    const auto restore = [&] {
      for (auto it = hidden.rbegin(); it != hidden.rend(); ++it) {
        std::filesystem::rename(it->second, it->first);
      }
      hidden.clear();
    };
    try {
      hide(config.stats_path);
      for (const auto& package : config.packages) {
        hide(std::filesystem::path(package) / "model.json");
      }
      request.dump_input_dir.clear();
      request.metadata_path.clear();
      request.standardized_output = (root / "second-standardized.f32").string();
      request.physical_output = (root / "second-physical.f32").string();
      const auto second = runner.run(request, CancellationToken{});
      check(first.point_count == second.point_count && first.block_count == second.block_count,
            "sequential request shape changed");
      check(expected == read_bytes(request.physical_output), "same seed must reproduce fields");
      check(Json::parse(first.metadata_json).at("session_load_count") == config.packages.size(),
            "incorrect initial load count");
      check(Json::parse(second.metadata_json).at("session_load_count") == config.packages.size(),
            "model sessions were reloaded");
      CancellationToken cancelled;
      cancelled.cancel();
      request.physical_output = (root / "cancelled-physical.f32").string();
      rejects([&] { (void)runner.run(request, cancelled); }, "native workflow cancelled");
      check(!std::filesystem::exists(request.physical_output), "cancelled request published output");
    } catch (...) {
      restore();
      throw;
    }
    restore();
    std::cout << "Persistent runner repeated-request fixture passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
