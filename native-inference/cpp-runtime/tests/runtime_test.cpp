#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::vector<std::byte> as_bytes(const std::vector<float>& values) {
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

class TemporaryPackage {
 public:
  TemporaryPackage(const std::string& manifest,
                   const std::vector<std::string>& artifacts,
                   const std::string& suffix = {}) {
    static std::size_t sequence = 0;
    const auto timestamp = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    root_ = std::filesystem::temp_directory_path() /
            ("pnmir-runtime-test-" + std::to_string(timestamp) + "-" +
             std::to_string(sequence++) + suffix);
    std::filesystem::create_directory(root_);
    std::ofstream manifest_stream(root_ / "model.json");
    manifest_stream << manifest;
    if (!manifest_stream) {
      throw std::runtime_error("failed to write temporary manifest");
    }
    manifest_stream.close();
    for (const auto& artifact : artifacts) {
      std::ofstream artifact_stream(root_ / artifact);
      artifact_stream << artifact;
      if (!artifact_stream) {
        throw std::runtime_error("failed to write temporary artifact");
      }
    }
  }

  TemporaryPackage(const TemporaryPackage&) = delete;
  TemporaryPackage& operator=(const TemporaryPackage&) = delete;

  ~TemporaryPackage() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  const std::filesystem::path& root() const { return root_; }

 private:
  std::filesystem::path root_;
};

class EmptySession final : public physicsnemo::inference::BackendSession {
 public:
  std::vector<physicsnemo::inference::OwnedTensor> run(
      const std::vector<physicsnemo::inference::TensorView>&) override {
    throw std::runtime_error("selection test session cannot run");
  }
};

class RecordingBackend final : public physicsnemo::inference::Backend {
 public:
  RecordingBackend(std::string name,
                   std::shared_ptr<std::filesystem::path> selected)
      : name_(std::move(name)), selected_(std::move(selected)) {}

  std::string name() const override { return name_; }

  bool supports(const physicsnemo::inference::ArtifactSpec& artifact,
                const physicsnemo::inference::SessionOptions& options) const override {
    const bool precision_matches =
        options.precision == "auto" || options.precision == artifact.precision;
    return artifact.backend == name_ && artifact.target == options.device.type &&
           precision_matches;
  }

  std::unique_ptr<physicsnemo::inference::BackendSession> create_session(
      const physicsnemo::inference::ModelPackage&, const physicsnemo::inference::ArtifactSpec& artifact,
      const physicsnemo::inference::SessionOptions&) const override {
    *selected_ = artifact.path;
    return std::make_unique<EmptySession>();
  }

 private:
  std::string name_;
  std::shared_ptr<std::filesystem::path> selected_;
};

constexpr const char* kMultipleArtifactsManifest = R"json({
  "format_version": 1,
  "model": {"name": "selection", "version": "1"},
  "inputs": [{"name": "input", "dtype": "float32", "shape": [1]}],
  "outputs": [{"name": "output", "dtype": "float32", "shape": [1]}],
  "artifacts": [
    {
      "backend": "first",
      "target": "cpu",
      "precision": "fp16",
      "path": "first-fp16.mock"
    },
    {
      "backend": "second",
      "target": "cpu",
      "precision": "fp32",
      "path": "second-fp32.mock"
    },
    {
      "backend": "first",
      "target": "cpu",
      "precision": "fp32",
      "path": "first-fp32.mock"
    }
  ]
})json";

constexpr const char* kLegacyPackageManifest = R"json({
  "format_version": 1,
  "model": {"name": "legacy", "version": "1"},
  "producer": {"name": "modelrt-export", "version": "0.1.0"},
  "inputs": [{"name": "input", "dtype": "float32", "shape": [1]}],
  "outputs": [{"name": "output", "dtype": "float32", "shape": [1]}],
  "artifacts": [{
    "backend": "mock",
    "target": "cpu",
    "precision": "fp32",
    "path": "legacy.mock"
  }]
})json";

void test_identity_package() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  check(package.manifest().model_name == "identity", "model name mismatch");
  check(package.manifest().artifacts.size() == 1, "artifact count mismatch");

  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
  auto session = runtime.create_session(package);
  check(session->backend_name() == "mock", "automatic backend selection failed");

  const std::vector<float> expected{1.0F, -2.0F, 3.5F};
  physicsnemo::inference::OwnedTensor input("input", physicsnemo::inference::DType::kFloat32, {}, {3},
                             as_bytes(expected));
  const auto outputs = session->run({input.view()});
  check(outputs.size() == 1, "output count mismatch");

  std::vector<float> actual(expected.size());
  const auto output = outputs.front().view();
  std::memcpy(actual.data(), output.data, output.byte_size);
  check(actual == expected, "mock identity changed values");
}

void test_legacy_rtpkg_package_remains_loadable() {
  const TemporaryPackage fixture(
      kLegacyPackageManifest, {"legacy.mock"}, ".rtpkg");
  const auto package = physicsnemo::inference::ModelPackage::load(fixture.root());
  check(package.manifest().model_name == "legacy",
        "legacy .rtpkg package did not load");
}

void test_shape_validation() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
  auto session = runtime.create_session(package);

  bool rejected = false;
  try {
    physicsnemo::inference::OwnedTensor input("input", physicsnemo::inference::DType::kFloat32, {}, {1, 2},
                               as_bytes({1.0F, 2.0F}));
    static_cast<void>(session->run({input.view()}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "rank mismatch was not rejected");
}

void test_tensor_byte_overflow_is_rejected_before_backend_execution() {
  using namespace physicsnemo::inference;
  std::string failures;
  const auto expect_overflow = [&](auto operation, const std::string& label) {
    try {
      operation();
      failures += label + " accepted overflowing storage; ";
    } catch (const std::overflow_error&) {
    } catch (const std::exception& error) {
      failures += label + " reached a later check: " + error.what() + "; ";
    }
  };
  for (const auto dtype : {DType::kFloat16, DType::kFloat32, DType::kInt64}) {
    const auto width = dtype_size(dtype);
    const auto limit = std::numeric_limits<std::size_t>::max() / width;
    // The element count fits size_t, but converting it to bytes wraps to
    // zero or a small nonzero allocation. No oversized allocation is needed.
    for (const std::size_t extra : {0U, 1U}) {
      const Shape shape{2, static_cast<std::int64_t>(limit / 2 + 1 + extra)};
      auto storage = std::make_shared<std::vector<std::byte>>(2 * extra * width);
      const std::string label = std::string(to_string(dtype)) +
                                " extra=" + std::to_string(extra) + " ";
      expect_overflow([&] {
        static_cast<void>(OwnedTensor("input", dtype, {}, shape, *storage));
      }, label + "OwnedTensor");
      expect_overflow([&] {
        static_cast<void>(SharedTensor("input", dtype, {}, shape,
                                       storage->data(), storage->size(), storage));
      }, label + "SharedTensor");

      ModelManifest manifest;
      manifest.inputs.push_back({"input", dtype, {-1, -1}});
      manifest.outputs.push_back({"output", dtype, {-1, -1}});
      InferenceSession session(manifest, std::make_unique<EmptySession>(), "probe");
      const TensorView input{
          "input", dtype, {}, shape, storage->data(), storage->size()};
      expect_overflow([&] {
        static_cast<void>(session.run({input}));
      }, label + "input view");
      OwnedTensor valid_input("input", dtype, {}, {2, 1},
                              std::vector<std::byte>(2 * width));
      const MutableTensorView output{
          "output", dtype, {}, shape, storage->data(), storage->size()};
      expect_overflow([&] {
        session.run_into({valid_input.view()}, {output});
      }, label + "output view");
    }
  }
  check(failures.empty(), failures);
}

void test_caller_owned_output() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
  auto session = runtime.create_session(package);

  const auto capabilities = session->capabilities();
  check(capabilities.accepts_cpu_inputs,
        "mock backend did not report CPU input support");
  check(capabilities.caller_owned_cpu_outputs,
        "mock backend did not report caller-owned CPU output support");
  check(!capabilities.accepts_device_inputs,
        "mock backend unexpectedly reported device input support");

  const std::vector<float> expected{1.0F, -2.0F, 3.5F};
  physicsnemo::inference::OwnedTensor input("input", physicsnemo::inference::DType::kFloat32, {}, {3},
                             as_bytes(expected));
  std::vector<float> actual(expected.size());
  physicsnemo::inference::MutableTensorView output{
      "output", physicsnemo::inference::DType::kFloat32, {}, {3}, actual.data(),
      actual.size() * sizeof(float)};
  session->run_into({input.view()}, {output});
  check(actual == expected, "caller-owned output changed identity values");
}

void test_backend_owned_output() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
  auto session = runtime.create_session(package);

  const std::vector<float> expected{1.0F, -2.0F, 3.5F};
  physicsnemo::inference::OwnedTensor input("input", physicsnemo::inference::DType::kFloat32, {}, {3},
                             as_bytes(expected));
  const auto outputs = session->run_owned({input.view()});
  check(outputs.size() == 1, "backend-owned output count mismatch");
  const auto output = outputs.front().view();
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), output.data, output.byte_size);
  check(output.device.type == physicsnemo::inference::DeviceType::kCpu,
        "default backend-owned output changed device");
  check(actual == expected, "backend-owned output changed identity values");
}

void test_caller_owned_output_validation() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
  auto session = runtime.create_session(package);

  physicsnemo::inference::OwnedTensor input("input", physicsnemo::inference::DType::kFloat32, {}, {3},
                             as_bytes({1.0F, 2.0F, 3.0F}));
  std::vector<float> storage(3);
  bool rejected = false;
  try {
    physicsnemo::inference::MutableTensorView output{
        "wrong", physicsnemo::inference::DType::kFloat32, {}, {3}, storage.data(),
        storage.size() * sizeof(float)};
    session->run_into({input.view()}, {output});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "unknown caller-owned output name was not rejected");

  rejected = false;
  try {
    physicsnemo::inference::MutableTensorView output{
        "output", physicsnemo::inference::DType::kFloat32, {}, {1, 2}, storage.data(),
        2 * sizeof(float)};
    session->run_into({input.view()}, {output});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "caller-owned output shape mismatch was not rejected");
}

void test_prepared_state_owns_external_storage() {
  auto storage = std::make_shared<std::vector<float>>(
      std::initializer_list<float>{1.0F, 2.0F, 3.0F});
  const std::weak_ptr<std::vector<float>> weak_storage = storage;
  std::shared_ptr<void> owner = storage;
  {
    physicsnemo::inference::PreparedState state(
        "geometry-v1",
        {physicsnemo::inference::SharedTensor(
            "prepared", physicsnemo::inference::DType::kFloat32, {}, {3}, storage->data(),
            storage->size() * sizeof(float), std::move(owner))});
    storage.reset();
    check(!weak_storage.expired(),
          "prepared state did not retain external storage");
    check(state.byte_size() == 3 * sizeof(float),
          "prepared state byte count is incorrect");
    const auto views = state.views();
    check(views.size() == 1 && views[0].name == "prepared",
          "prepared state did not expose its tensor");
  }
  check(weak_storage.expired(),
        "prepared state did not release external storage");
}

void test_missing_backend() {
  const auto path =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());

  physicsnemo::inference::SessionOptions options;
  options.backend = "missing";
  bool rejected = false;
  try {
    static_cast<void>(runtime.create_session(package, options));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected, "missing backend was not rejected");
}

void test_manifest_order_selects_first_compatible_artifact() {
  const TemporaryPackage fixture(
      kMultipleArtifactsManifest,
      {"first-fp16.mock", "second-fp32.mock", "first-fp32.mock"});
  const auto package = physicsnemo::inference::ModelPackage::load(fixture.root());
  const auto first_selected = std::make_shared<std::filesystem::path>();
  const auto second_selected = std::make_shared<std::filesystem::path>();

  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(
      std::make_unique<RecordingBackend>("second", second_selected));
  runtime.register_backend(
      std::make_unique<RecordingBackend>("first", first_selected));
  const auto session = runtime.create_session(package);

  check(session->backend_name() == "first",
        "manifest order did not select the first compatible backend");
  check(*first_selected == "first-fp16.mock",
        "manifest order did not select the first compatible artifact");
  check(second_selected->empty(), "later backend was unexpectedly created");
}

void test_explicit_backend_and_precision_filter_artifacts() {
  const TemporaryPackage fixture(
      kMultipleArtifactsManifest,
      {"first-fp16.mock", "second-fp32.mock", "first-fp32.mock"});
  const auto package = physicsnemo::inference::ModelPackage::load(fixture.root());
  const auto first_selected = std::make_shared<std::filesystem::path>();
  const auto second_selected = std::make_shared<std::filesystem::path>();

  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(
      std::make_unique<RecordingBackend>("first", first_selected));
  runtime.register_backend(
      std::make_unique<RecordingBackend>("second", second_selected));
  physicsnemo::inference::SessionOptions options;
  options.backend = "first";
  options.precision = "fp32";
  const auto session = runtime.create_session(package, options);

  check(session->backend_name() == "first",
        "explicit backend selection was ignored");
  check(*first_selected == "first-fp32.mock",
        "precision did not filter incompatible artifacts");
  check(second_selected->empty(),
        "unrequested backend was unexpectedly created");
}

void test_required_operator_abi_is_enforced() {
  constexpr const char* manifest = R"json({
    "format_version": 1,
    "model": {"name": "custom-op", "version": "1"},
    "inputs": [{"name": "input", "dtype": "float32", "shape": [1]}],
    "outputs": [{"name": "output", "dtype": "float32", "shape": [1]}],
    "artifacts": [{
      "backend": "custom",
      "target": "cpu",
      "precision": "fp32",
      "path": "custom.mock",
      "required_operators": [{
        "id": "example.operator",
        "abi": "v1"
      }]
    }]
  })json";
  const TemporaryPackage fixture(manifest, {"custom.mock"});
  const auto package = physicsnemo::inference::ModelPackage::load(fixture.root());
  check(package.manifest().artifacts.front().required_operators.size() == 1,
        "required operator was not parsed");

  const auto create_runtime = [] {
    physicsnemo::inference::Runtime runtime;
    runtime.register_backend(std::make_unique<RecordingBackend>(
        "custom", std::make_shared<std::filesystem::path>()));
    return runtime;
  };

  auto missing = create_runtime();
  bool rejected_missing = false;
  try {
    static_cast<void>(missing.create_session(package));
  } catch (const std::runtime_error&) {
    rejected_missing = true;
  }
  check(rejected_missing, "missing required operator was not rejected");

  auto wrong_abi = create_runtime();
  wrong_abi.register_operator("example.operator", "v2");
  bool rejected_abi = false;
  try {
    static_cast<void>(wrong_abi.create_session(package));
  } catch (const std::runtime_error&) {
    rejected_abi = true;
  }
  check(rejected_abi, "wrong required operator ABI was not rejected");

  auto compatible = create_runtime();
  compatible.register_operator("example.operator", "v1");
  const auto session = compatible.create_session(package);
  check(session->backend_name() == "custom",
        "matching required operator ABI was not accepted");
}

void test_package_rejects_parent_artifact_path() {
  constexpr const char* manifest = R"json({
    "format_version": 1,
    "model": {"name": "unsafe", "version": "1"},
    "inputs": [{"name": "input", "dtype": "float32", "shape": [1]}],
    "outputs": [{"name": "output", "dtype": "float32", "shape": [1]}],
    "artifacts": [{
      "backend": "mock",
      "target": "cpu",
      "precision": "fp32",
      "path": "../outside.mock"
    }]
  })json";
  const TemporaryPackage fixture(manifest, {});

  bool rejected = false;
  try {
    static_cast<void>(physicsnemo::inference::ModelPackage::load(fixture.root()));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "parent artifact path was not rejected");
}

}  // namespace

int main() {
  try {
    test_identity_package();
    test_legacy_rtpkg_package_remains_loadable();
    test_shape_validation();
    test_tensor_byte_overflow_is_rejected_before_backend_execution();
    test_caller_owned_output();
    test_backend_owned_output();
    test_caller_owned_output_validation();
    test_prepared_state_owns_external_storage();
    test_missing_backend();
    test_manifest_order_selects_first_compatible_artifact();
    test_explicit_backend_and_precision_filter_artifacts();
    test_required_operator_abi_is_enforced();
    test_package_rejects_parent_artifact_path();
    std::cout << "all tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
