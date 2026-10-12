#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/v1/api.hpp"

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> copy_values(const physicsnemo::inference::TensorView& tensor) {
  std::vector<float> values(tensor.byte_size / sizeof(float));
  std::memcpy(values.data(), tensor.data, tensor.byte_size);
  return values;
}

std::vector<std::byte> copy_bytes(const std::vector<float>& values) {
  std::vector<std::byte> storage(values.size() * sizeof(float));
  std::memcpy(storage.data(), values.data(), storage.size());
  return storage;
}

physicsnemo::inference::TensorView input_view(const std::vector<float>& values) {
  return {"input", physicsnemo::inference::DType::kFloat32, {}, {3}, values.data(),
          values.size() * sizeof(float)};
}

physicsnemo::inference::v1::Executor make_mock_executor() {
  physicsnemo::inference::v1::Engine engine;
  engine.register_backend(physicsnemo::inference::create_mock_backend());
  const auto package =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  return engine.load_model(package).create_executor();
}

void test_runtime_owned_result() {
  auto executor = make_mock_executor();
  const std::vector<float> expected{1.0F, -2.0F, 3.5F};
  physicsnemo::inference::v1::Request request;
  request.bind_input(input_view(expected));

  const auto result = executor.run(request);

  check(result.output_count() == 1, "runtime-owned output count mismatch");
  check(copy_values(result.output("output")) == expected,
        "runtime-owned execution changed identity values");
  check(executor.backend_name() == "mock", "backend name was not exposed");
}

void test_caller_owned_result() {
  auto executor = make_mock_executor();
  auto input = std::make_shared<std::vector<float>>(
      std::initializer_list<float>{1.0F, -2.0F, 3.5F});
  auto output = std::make_shared<std::vector<float>>(input->size());
  std::weak_ptr<std::vector<float>> weak_output = output;

  physicsnemo::inference::v1::Request request;
  request
      .bind_input(input_view(*input), input)
      .bind_output(
          {"output", physicsnemo::inference::DType::kFloat32, {}, {3}, output->data(),
           output->size() * sizeof(float)},
          output);

  auto result = executor.run(request);
  output.reset();

  check(!weak_output.expired(),
        "result did not retain caller-owned output storage");
  check(copy_values(result.output(0)) == *input,
        "caller-owned execution changed identity values");
}

struct BlockingState {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered{false};
  bool released{false};
};

class BlockingSession final : public physicsnemo::inference::BackendSession {
 public:
  explicit BlockingSession(std::shared_ptr<BlockingState> state)
      : state_(std::move(state)) {}

  std::vector<physicsnemo::inference::OwnedTensor> run(
      const std::vector<physicsnemo::inference::TensorView>& inputs) override {
    {
      std::unique_lock lock(state_->mutex);
      state_->entered = true;
      state_->changed.notify_all();
      state_->changed.wait(lock, [this] { return state_->released; });
    }

    const auto& input = inputs.front();
    std::vector<std::byte> storage(input.byte_size);
    std::memcpy(storage.data(), input.data, input.byte_size);
    return {physicsnemo::inference::OwnedTensor("output", input.dtype, input.device,
                               input.shape, std::move(storage))};
  }

 private:
  std::shared_ptr<BlockingState> state_;
};

class BlockingBackend final : public physicsnemo::inference::Backend {
 public:
  explicit BlockingBackend(std::shared_ptr<BlockingState> state)
      : state_(std::move(state)) {}

  std::string name() const override { return "mock"; }

  bool supports(const physicsnemo::inference::ArtifactSpec& artifact,
                const physicsnemo::inference::SessionOptions& options) const override {
    return artifact.backend == name() &&
           artifact.target == options.device.type;
  }

  std::unique_ptr<physicsnemo::inference::BackendSession> create_session(
      const physicsnemo::inference::ModelPackage&, const physicsnemo::inference::ArtifactSpec&,
      const physicsnemo::inference::SessionOptions&) const override {
    return std::make_unique<BlockingSession>(state_);
  }

 private:
  std::shared_ptr<BlockingState> state_;
};

struct ConcurrentState {
  std::atomic<int> active{0};
  std::atomic<int> maximum{0};
};

class ConcurrentSession final : public physicsnemo::inference::BackendSession {
 public:
  explicit ConcurrentSession(std::shared_ptr<ConcurrentState> state)
      : state_(std::move(state)) {}

  std::vector<physicsnemo::inference::OwnedTensor> run(
      const std::vector<physicsnemo::inference::TensorView>& inputs) override {
    const int active = state_->active.fetch_add(1) + 1;
    int maximum = state_->maximum.load();
    while (active > maximum &&
           !state_->maximum.compare_exchange_weak(maximum, active)) {
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    state_->active.fetch_sub(1);

    const auto& input = inputs.front();
    std::vector<std::byte> storage(input.byte_size);
    std::memcpy(storage.data(), input.data, input.byte_size);
    return {physicsnemo::inference::OwnedTensor("output", input.dtype, input.device,
                               input.shape, std::move(storage))};
  }

 private:
  std::shared_ptr<ConcurrentState> state_;
};

class ConcurrentBackend final : public physicsnemo::inference::Backend {
 public:
  explicit ConcurrentBackend(std::shared_ptr<ConcurrentState> state)
      : state_(std::move(state)) {}

  std::string name() const override { return "mock"; }

  bool supports(const physicsnemo::inference::ArtifactSpec& artifact,
                const physicsnemo::inference::SessionOptions& options) const override {
    return artifact.backend == name() &&
           artifact.target == options.device.type;
  }

  std::unique_ptr<physicsnemo::inference::BackendSession> create_session(
      const physicsnemo::inference::ModelPackage&, const physicsnemo::inference::ArtifactSpec&,
      const physicsnemo::inference::SessionOptions&) const override {
    return std::make_unique<ConcurrentSession>(state_);
  }

 private:
  std::shared_ptr<ConcurrentState> state_;
};

void test_async_retains_input_until_completion() {
  auto state = std::make_shared<BlockingState>();
  physicsnemo::inference::v1::Engine engine;
  engine.register_backend(std::make_unique<BlockingBackend>(state));
  const auto package =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  auto executor = engine.load_model(package).create_executor();

  std::weak_ptr<std::vector<float>> weak_input;
  physicsnemo::inference::v1::Ticket ticket;
  {
    auto input = std::make_shared<std::vector<float>>(
        std::initializer_list<float>{1.0F, -2.0F, 3.5F});
    weak_input = input;
    physicsnemo::inference::v1::Request request;
    request.bind_input(input_view(*input), input);
    ticket = executor.submit(std::move(request));
  }

  {
    std::unique_lock lock(state->mutex);
    check(state->changed.wait_for(
              lock, std::chrono::seconds(1),
              [&state] { return state->entered; }),
          "asynchronous execution did not start");
  }
  check(!weak_input.expired(),
        "asynchronous request did not retain input storage");

  {
    std::lock_guard lock(state->mutex);
    state->released = true;
    state->changed.notify_all();
  }

  const auto result = ticket.get();
  check(ticket.ready(), "completed ticket did not become ready");
  check(copy_values(result.output("output")) ==
            std::vector<float>({1.0F, -2.0F, 3.5F}),
        "asynchronous execution changed identity values");
}

void test_async_rejects_borrowed_input() {
  auto executor = make_mock_executor();
  const std::vector<float> input{1.0F, -2.0F, 3.5F};
  physicsnemo::inference::v1::Request request;
  request.bind_input(input_view(input));

  bool rejected = false;
  try {
    static_cast<void>(executor.submit(std::move(request)));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "asynchronous execution accepted borrowed input storage");
}

void test_executor_serializes_backend_session_calls() {
  auto state = std::make_shared<ConcurrentState>();
  physicsnemo::inference::v1::Engine engine;
  engine.register_backend(std::make_unique<ConcurrentBackend>(state));
  const auto package =
      std::filesystem::path(PNMIR_SOURCE_DIR) / "tests" / "fixtures" / "identity";
  auto executor = engine.load_model(package).create_executor();

  std::vector<physicsnemo::inference::v1::Ticket> tickets;
  for (int index = 0; index < 4; ++index) {
    const std::vector<float> values{
        static_cast<float>(index), 2.0F, 3.0F};
    physicsnemo::inference::v1::Request request;
    request.bind_input(physicsnemo::inference::OwnedTensor(
        "input", physicsnemo::inference::DType::kFloat32, {}, {3}, copy_bytes(values)));
    tickets.push_back(executor.submit(std::move(request)));
  }

  for (const auto& ticket : tickets) {
    check(ticket.get().output_count() == 1,
          "serialized asynchronous execution did not produce an output");
  }
  check(state->maximum.load() == 1,
        "executor called a backend session concurrently");
}

}  // namespace

int main() {
  try {
    test_runtime_owned_result();
    test_caller_owned_result();
    test_async_retains_input_until_completion();
    test_async_rejects_borrowed_input();
    test_executor_serializes_backend_session_calls();
    std::cout << "all v1 API tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
