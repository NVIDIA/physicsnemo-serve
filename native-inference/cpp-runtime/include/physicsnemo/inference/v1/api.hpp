#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/prepared_state.hpp"
#include "physicsnemo/inference/tensor.hpp"

namespace physicsnemo::inference {

class Backend;

namespace v1 {

inline constexpr int kApiVersion = 1;

struct ExecutorOptions {
  // Empty selects the first compatible backend in manifest order.
  std::string backend;
  Device device{};
  std::string precision{"auto"};
};

struct ExecutorCapabilities {
  bool accepts_cpu_inputs{true};
  bool accepts_device_inputs{false};
  bool caller_owned_cpu_outputs{true};
  bool caller_owned_device_outputs{false};
  bool backend_owned_device_outputs{false};
};

class Model;
class Executor;

class Engine {
 public:
  Engine();
  ~Engine();

  Engine(const Engine&) = default;
  Engine& operator=(const Engine&) = default;
  Engine(Engine&&) noexcept = default;
  Engine& operator=(Engine&&) noexcept = default;

  // Backend registration is an engine-construction concern. Model execution
  // stays backend-neutral once an Executor has been created.
  void register_backend(std::unique_ptr<Backend> backend);
  void register_operator(std::string id, std::string abi);
  Model load_model(const std::filesystem::path& package_path) const;

 private:
  struct State;
  std::shared_ptr<State> state_;

  friend class Model;
  friend class Executor;
};

class Model {
 public:
  Model() = default;

  explicit operator bool() const noexcept { return impl_ != nullptr; }
  const ModelManifest& manifest() const;
  Executor create_executor(const ExecutorOptions& options = {}) const;

 private:
  struct Impl;
  explicit Model(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class Engine;
};

class Request {
 public:
  Request();
  ~Request();

  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
  Request(Request&&) noexcept;
  Request& operator=(Request&&) noexcept;

  // Borrowed inputs are valid only for a blocking run().
  Request& bind_input(TensorView input);

  // The owner keeps the allocation containing input.data alive.
  Request& bind_input(TensorView input,
                      std::shared_ptr<const void> owner);
  Request& bind_input(OwnedTensor input);
  Request& bind_input(SharedTensor input);

  // Caller-bound outputs require an owner because Result retains and exposes
  // their storage after execution returns.
  Request& bind_output(MutableTensorView output,
                       std::shared_ptr<void> owner);

  std::size_t input_count() const noexcept;
  std::size_t output_count() const noexcept;

 private:
  bool async_safe() const noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Executor;
};

class Result {
 public:
  Result();

  std::size_t output_count() const noexcept;
  TensorView output(std::size_t index) const;
  TensorView output(std::string_view name) const;
  std::vector<TensorView> output_views() const;

 private:
  struct Impl;
  explicit Result(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;

  friend class Executor;
};

class Ticket {
 public:
  Ticket() = default;

  explicit operator bool() const noexcept { return impl_ != nullptr; }
  bool ready() const;
  void wait() const;
  Result get() const;

 private:
  struct Impl;
  explicit Ticket(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class Executor;
};

class Executor {
 public:
  Executor() = default;

  explicit operator bool() const noexcept { return impl_ != nullptr; }

  // Calls on the same Executor are safe from multiple host threads. The v1
  // implementation serializes access to its backend session.
  Result run(const Request& request) const;

  // Asynchronous submission requires owned input bindings. The returned
  // Ticket retains the Executor, Request metadata, and allocation owners.
  Ticket submit(Request&& request) const;

  ExecutorCapabilities capabilities() const;
  std::string backend_name() const;

 private:
  struct Impl;
  explicit Executor(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;

  friend class Model;
};

}  // namespace v1
}  // namespace physicsnemo::inference
