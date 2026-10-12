#include "physicsnemo/inference/v1/api.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/runtime.hpp"

namespace physicsnemo::inference::v1 {
namespace {

template <typename Binding>
void ensure_name_is_available(
    std::string_view name, const std::vector<Binding>& bindings,
    std::string_view kind) {
  const auto duplicate =
      std::ranges::find_if(bindings, [name](const auto& binding) {
        return binding.view.name == name;
      });
  if (duplicate != bindings.end()) {
    throw std::invalid_argument(std::string(kind) + " is already bound: " +
                                std::string(name));
  }
}

void ensure_valid_name(std::string_view name, std::string_view kind) {
  if (name.empty()) {
    throw std::invalid_argument(std::string(kind) +
                                " binding name cannot be empty");
  }
}

}  // namespace

struct Engine::State {
  Runtime runtime;
  mutable std::mutex mutex;
};

struct Model::Impl {
  std::shared_ptr<Engine::State> engine;
  ModelPackage package;
};

struct Request::Impl {
  struct InputBinding {
    TensorView view;
    std::shared_ptr<const void> owner;
  };

  struct OutputBinding {
    MutableTensorView view;
    std::shared_ptr<void> owner;
  };

  std::vector<InputBinding> inputs;
  std::vector<OutputBinding> outputs;
};

struct Result::Impl {
  explicit Impl(std::vector<SharedTensor> result_outputs)
      : outputs(std::move(result_outputs)) {}

  std::vector<SharedTensor> outputs;
};

struct Executor::Impl {
  Impl(std::shared_ptr<Engine::State> engine_state,
       std::unique_ptr<InferenceSession> inference_session)
      : engine(std::move(engine_state)),
        session(std::move(inference_session)) {}

  Result execute(const Request& request) {
    std::lock_guard lock(mutex);

    std::vector<TensorView> inputs;
    inputs.reserve(request.impl_->inputs.size());
    for (const auto& input : request.impl_->inputs) {
      inputs.push_back(input.view);
    }

    if (request.impl_->outputs.empty()) {
      return Result(
          std::make_shared<Result::Impl>(session->run_owned(inputs)));
    }

    std::vector<MutableTensorView> output_views;
    output_views.reserve(request.impl_->outputs.size());
    for (const auto& output : request.impl_->outputs) {
      output_views.push_back(output.view);
    }
    session->run_into(inputs, output_views);

    std::vector<SharedTensor> outputs;
    outputs.reserve(request.impl_->outputs.size());
    for (const auto& output : request.impl_->outputs) {
      const auto view = output.view.as_read_only();
      outputs.emplace_back(view.name, view.dtype, view.device, view.shape,
                           view.data, view.byte_size, output.owner);
    }
    return Result(
        std::make_shared<Result::Impl>(std::move(outputs)));
  }

  ExecutorCapabilities get_capabilities() {
    std::lock_guard lock(mutex);
    const auto supported = session->capabilities();
    return {
        supported.accepts_cpu_inputs,
        supported.accepts_device_inputs,
        supported.caller_owned_cpu_outputs,
        supported.caller_owned_device_outputs,
        supported.backend_owned_device_outputs,
    };
  }

  std::string get_backend_name() {
    std::lock_guard lock(mutex);
    return session->backend_name();
  }

  // Keep backend registrations alive for the entire session lifetime.
  std::shared_ptr<Engine::State> engine;
  std::unique_ptr<InferenceSession> session;
  std::mutex mutex;
};

struct Ticket::Impl {
  explicit Impl(std::shared_future<Result> submitted)
      : future(std::move(submitted)) {}

  std::shared_future<Result> future;
};

Engine::Engine() : state_(std::make_shared<State>()) {}

Engine::~Engine() = default;

void Engine::register_backend(std::unique_ptr<Backend> backend) {
  if (state_ == nullptr) {
    throw std::logic_error("engine is not initialized");
  }
  std::lock_guard lock(state_->mutex);
  state_->runtime.register_backend(std::move(backend));
}

void Engine::register_operator(std::string id, std::string abi) {
  if (state_ == nullptr) {
    throw std::logic_error("engine is not initialized");
  }
  std::lock_guard lock(state_->mutex);
  state_->runtime.register_operator(std::move(id), std::move(abi));
}

Model Engine::load_model(const std::filesystem::path& package_path) const {
  if (state_ == nullptr) {
    throw std::logic_error("engine is not initialized");
  }
  return Model(std::make_shared<Model::Impl>(
      Model::Impl{state_, ModelPackage::load(package_path)}));
}

Model::Model(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

const ModelManifest& Model::manifest() const {
  if (impl_ == nullptr) {
    throw std::logic_error("model is not initialized");
  }
  return impl_->package.manifest();
}

Executor Model::create_executor(const ExecutorOptions& options) const {
  if (impl_ == nullptr) {
    throw std::logic_error("model is not initialized");
  }

  SessionOptions session_options;
  if (!options.backend.empty()) {
    session_options.backend = options.backend;
  }
  session_options.device = options.device;
  session_options.precision = options.precision;

  std::lock_guard lock(impl_->engine->mutex);
  auto session =
      impl_->engine->runtime.create_session(impl_->package, session_options);
  return Executor(std::make_shared<Executor::Impl>(
      impl_->engine, std::move(session)));
}

Request::Request() : impl_(std::make_unique<Impl>()) {}

Request::~Request() = default;

Request::Request(Request&&) noexcept = default;

Request& Request::operator=(Request&&) noexcept = default;

Request& Request::bind_input(TensorView input) {
  if (impl_ == nullptr) {
    throw std::logic_error("request is not initialized");
  }
  ensure_valid_name(input.name, "input");
  ensure_name_is_available(input.name, impl_->inputs, "input");
  impl_->inputs.push_back({std::move(input), {}});
  return *this;
}

Request& Request::bind_input(TensorView input,
                             std::shared_ptr<const void> owner) {
  if (impl_ == nullptr) {
    throw std::logic_error("request is not initialized");
  }
  if (input.byte_size != 0 && owner == nullptr) {
    throw std::invalid_argument(
        "owned input binding requires an allocation owner");
  }
  ensure_valid_name(input.name, "input");
  ensure_name_is_available(input.name, impl_->inputs, "input");
  impl_->inputs.push_back({std::move(input), std::move(owner)});
  return *this;
}

Request& Request::bind_input(OwnedTensor input) {
  auto owner = std::make_shared<OwnedTensor>(std::move(input));
  const auto view = owner->view();
  return bind_input(view, std::move(owner));
}

Request& Request::bind_input(SharedTensor input) {
  auto owner = std::make_shared<SharedTensor>(std::move(input));
  const auto view = owner->view();
  return bind_input(view, std::move(owner));
}

Request& Request::bind_output(MutableTensorView output,
                              std::shared_ptr<void> owner) {
  if (impl_ == nullptr) {
    throw std::logic_error("request is not initialized");
  }
  if (output.byte_size != 0 && owner == nullptr) {
    throw std::invalid_argument(
        "output binding requires an allocation owner");
  }
  ensure_valid_name(output.name, "output");
  ensure_name_is_available(output.name, impl_->outputs, "output");
  impl_->outputs.push_back({std::move(output), std::move(owner)});
  return *this;
}

bool Request::async_safe() const noexcept {
  return impl_ != nullptr &&
         std::ranges::all_of(impl_->inputs, [](const auto& input) {
           return input.view.byte_size == 0 || input.owner != nullptr;
         });
}

std::size_t Request::input_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->inputs.size();
}

std::size_t Request::output_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->outputs.size();
}

Result::Result()
    : impl_(std::make_shared<Impl>(std::vector<SharedTensor>{})) {}

Result::Result(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

std::size_t Result::output_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->outputs.size();
}

TensorView Result::output(std::size_t index) const {
  if (impl_ == nullptr || index >= impl_->outputs.size()) {
    throw std::out_of_range("result output index is out of range");
  }
  return impl_->outputs[index].view();
}

TensorView Result::output(std::string_view name) const {
  if (impl_ == nullptr) {
    throw std::out_of_range("result has no output named: " +
                            std::string(name));
  }
  const auto found = std::ranges::find_if(
      impl_->outputs, [name](const auto& output_tensor) {
        return output_tensor.view().name == name;
      });
  if (found == impl_->outputs.end()) {
    throw std::out_of_range("result has no output named: " +
                            std::string(name));
  }
  return found->view();
}

std::vector<TensorView> Result::output_views() const {
  std::vector<TensorView> views;
  if (impl_ == nullptr) {
    return views;
  }
  views.reserve(impl_->outputs.size());
  for (const auto& output_tensor : impl_->outputs) {
    views.push_back(output_tensor.view());
  }
  return views;
}

Ticket::Ticket(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool Ticket::ready() const {
  if (impl_ == nullptr) {
    throw std::logic_error("ticket is not initialized");
  }
  return impl_->future.wait_for(std::chrono::seconds(0)) ==
         std::future_status::ready;
}

void Ticket::wait() const {
  if (impl_ == nullptr) {
    throw std::logic_error("ticket is not initialized");
  }
  impl_->future.wait();
}

Result Ticket::get() const {
  if (impl_ == nullptr) {
    throw std::logic_error("ticket is not initialized");
  }
  return impl_->future.get();
}

Executor::Executor(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result Executor::run(const Request& request) const {
  if (impl_ == nullptr) {
    throw std::logic_error("executor is not initialized");
  }
  if (request.impl_ == nullptr) {
    throw std::logic_error("request is not initialized");
  }
  return impl_->execute(request);
}

Ticket Executor::submit(Request&& request) const {
  if (impl_ == nullptr) {
    throw std::logic_error("executor is not initialized");
  }
  if (request.impl_ == nullptr) {
    throw std::logic_error("request is not initialized");
  }
  if (!request.async_safe()) {
    throw std::invalid_argument(
        "asynchronous inputs require allocation owners");
  }

  auto future = std::async(
      std::launch::async,
      [impl = impl_, submitted = std::move(request)]() mutable {
        return impl->execute(submitted);
      });
  return Ticket(std::make_shared<Ticket::Impl>(std::move(future).share()));
}

ExecutorCapabilities Executor::capabilities() const {
  if (impl_ == nullptr) {
    throw std::logic_error("executor is not initialized");
  }
  return impl_->get_capabilities();
}

std::string Executor::backend_name() const {
  if (impl_ == nullptr) {
    throw std::logic_error("executor is not initialized");
  }
  return impl_->get_backend_name();
}

}  // namespace physicsnemo::inference::v1
