#include "physicsnemo/workflow/v1/host.hpp"
#include "physicsnemo/workflow/v1/authoring.hpp"
#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/v1/self_test.hpp"
#ifdef PNMS_HAS_AOTI
#include "physicsnemo/inference/backends/aoti.hpp"
#endif
#ifdef PNMS_HAS_TENSORRT
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif
#ifdef PNMS_HAS_ONNXRUNTIME
#include "physicsnemo/inference/backends/onnxruntime.hpp"
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

namespace physicsnemo::workflow::v1 {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
namespace sdk = physicsnemo::inference;
void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument(message);
}
void cancelled(const std::function<bool()>& callback) {
  if (callback && callback()) throw std::runtime_error("native workflow cancelled");
}
bool identifier(const std::string& name) {
  return !name.empty() && name.size() <= 128 &&
      std::ranges::all_of(name, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-';
      });
}
void exact_names(const Json& object, const std::vector<sdk::TensorSpec>& specs,
                 const std::string& context) {
  require(object.is_object() && object.size() == specs.size(),
          context + " must specify exactly the declared tensor names");
  for (const auto& spec : specs)
    require(object.contains(spec.name), context + " missing tensor: " + spec.name);
}
std::size_t static_count(const sdk::TensorSpec& spec, std::size_t maximum) {
  require(spec.dtype == sdk::DType::kFloat32, "workflow supports static float32 tensors only: " + spec.name);
  require(std::ranges::all_of(spec.shape, [](auto n) { return n > 0; }),
          "workflow requires positive static tensor dimensions: " + spec.name);
  const auto bytes = sdk::tensor_byte_size(spec.shape, spec.dtype);
  require(bytes <= maximum, "tensor exceeds max_tensor_bytes: " + spec.name);
  return bytes / sizeof(float);
}
std::vector<float> json_values(const Json& value, const sdk::TensorSpec& spec,
                               std::size_t maximum) {
  const auto count = static_count(spec, maximum);
  require(value.is_array() && value.size() == count, "tensor element count differs: " + spec.name);
  std::vector<float> values;
  values.reserve(count);
  for (const auto& item : value) {
    require(item.is_number() && !item.is_boolean(), "tensor values must be numeric: " + spec.name);
    const auto number = item.get<double>();
    const auto converted = static_cast<float>(number);
    require(std::isfinite(number) && std::isfinite(converted),
            "tensor values must be finite float32: " + spec.name);
    values.push_back(converted);
  }
  return values;
}
std::vector<std::byte> bytes_of(const std::vector<float>& values) {
  std::vector<std::byte> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
std::vector<float> output_values(const sdk::TensorView& view, const sdk::TensorSpec& spec,
                                 std::size_t maximum) {
  const auto count = static_count(spec, maximum);
  require(view.name == spec.name && view.dtype == sdk::DType::kFloat32 && view.shape == spec.shape,
          "model returned the wrong output signature: " + spec.name);
  require(view.device.type == sdk::DeviceType::kCpu && view.byte_size == count * sizeof(float) && view.data,
          "model output must own complete CPU float32 storage: " + spec.name);
  std::vector<float> result(count);
  std::memcpy(result.data(), view.data, view.byte_size);
  require(std::ranges::all_of(result, [](float n) { return std::isfinite(n); }),
          "model output must be finite: " + spec.name);
  return result;
}
void no_symlink(const fs::path& path) {
  auto current = fs::path{};
  for (const auto& part : fs::absolute(path)) {
    current /= part;
    require(!fs::is_symlink(fs::symlink_status(current)),
            "symlink path is not admitted: " + path.string());
  }
}
fs::path regular_path(const std::string& raw) {
  fs::path path(raw);
  require(path.is_absolute(), "admitted paths must be absolute");
  no_symlink(path);
  require(fs::is_regular_file(path), "admitted input/asset must be a regular file: " + raw);
  return fs::canonical(path);
}
struct ResidentModel {
  sdk::v1::Model model;
  sdk::v1::Executor executor;
  Json fixture;
  std::string package_path;
};
class Models {
 public:
  std::map<std::string, ResidentModel> entries;
  std::size_t maximum{67108864};
  const ResidentModel& get(const std::string& alias) const {
    const auto found = entries.find(alias);
    require(found != entries.end(), "unknown fixed model alias: " + alias);
    return found->second;
  }
  sdk::v1::Result infer(const std::string& alias, const std::vector<sdk::TensorView>& inputs) const {
    const auto& resident = get(alias);
    const auto& manifest = resident.model.manifest();
    require(inputs.size() == manifest.inputs.size(), "model input count differs");
    sdk::v1::Request request;
    std::set<std::string> seen;
    for (const auto& input : inputs) {
      const auto spec = std::ranges::find_if(manifest.inputs, [&](const auto& s) { return s.name == input.name; });
      require(spec != manifest.inputs.end() && seen.insert(input.name).second,
              "unknown or duplicate model input: " + input.name);
      const auto count = static_count(*spec, maximum);
      require(input.dtype == sdk::DType::kFloat32 && input.shape == spec->shape &&
              input.device.type == sdk::DeviceType::kCpu &&
              input.byte_size == count * sizeof(float) && input.data,
              "model input signature/element count differs: " + input.name);
      const auto* values = static_cast<const float*>(input.data);
      require(std::all_of(values, values + count, [](float n) { return std::isfinite(n); }),
              "model input must be finite: " + input.name);
      request.bind_input(input);
    }
    for (const auto& output : manifest.outputs) {
      auto storage = std::make_shared<std::vector<std::byte>>(static_count(output, maximum) * sizeof(float));
      request.bind_output({output.name, output.dtype, {}, output.shape,
                           storage->data(), storage->size()}, storage);
    }
    auto result = resident.executor.run(request);
    require(result.output_count() == manifest.outputs.size(), "model returned wrong output count");
    for (const auto& output : manifest.outputs)
      (void)output_values(result.output(output.name), output, maximum);
    return result;
  }
};
}

class Context::Impl {
 public:
  Models* models;
  Json inputs, assets;
  fs::path root;
  Json artifacts{Json::array()};
  std::function<bool()> cancellation;
  const std::string* configuration;
  Json input_metadata;
};
class InitializationContext::Impl {
 public:
  Models* models;
  const Json* assets;
  const std::string* configuration;
  const std::function<bool()>* cancellation;
};
const std::string& InitializationContext::configuration_json() const {
  return *impl_->configuration;
}
const sdk::ModelManifest& InitializationContext::model(const std::string& alias) const {
  return impl_->models->get(alias).model.manifest();
}
fs::path InitializationContext::asset(const std::string& name) const {
  require(impl_->assets->contains(name) && impl_->assets->at(name).is_string(),
          "unknown configured asset: " + name);
  return regular_path(impl_->assets->at(name).get<std::string>());
}
void InitializationContext::check_cancelled() const { cancelled(*impl_->cancellation); }
const std::string& Context::configuration_json() const {
  return *impl_->configuration;
}
sdk::v1::Result Context::infer(const std::string& alias,
                              const std::vector<sdk::TensorView>& inputs) {
  check_cancelled();
  auto result = impl_->models->infer(alias, inputs);
  check_cancelled();
  return result;
}
const sdk::ModelManifest& Context::model(const std::string& alias) const {
  return impl_->models->get(alias).model.manifest();
}
void Context::check_cancelled() const { cancelled(impl_->cancellation); }
fs::path Context::input(const std::string& name) const {
  require(impl_->inputs.contains(name) && impl_->inputs.at(name).is_string(),
          "unknown admitted input: " + name);
  return regular_path(impl_->inputs.at(name).get<std::string>());
}
AdmittedInput Context::input_info(const std::string& name) const {
  require(impl_->input_metadata.contains(name), "verified input metadata is unavailable: " + name);
  const auto& info = impl_->input_metadata.at(name);
  require(info.is_object() && info.size() == 5 && info.contains("artifact_id") &&
          info.contains("size_bytes") && info.contains("sha256") &&
          info.contains("media_type") && info.contains("source"), "invalid input metadata fields: " + name);
  const auto id = info.at("artifact_id").get<std::string>();
  const auto digest = info.at("sha256").get<std::string>();
  const auto media = info.at("media_type").get<std::string>();
  const auto source = info.at("source").get<std::string>();
  require(!id.empty() && id.size() <= 256, "invalid input metadata artifact_id: " + name);
  require(digest.size() == 64 && std::ranges::all_of(digest, [](unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  }), "invalid input metadata sha256: " + name);
  require(!media.empty() && media.size() <= 128 && media.find_first_of("\r\n") == std::string::npos,
          "invalid input metadata media_type: " + name);
  require(source == "request" || source == "release", "invalid input metadata source: " + name);
  require(info.at("size_bytes").is_number_unsigned(), "input metadata size_bytes must be uint64: " + name);
  const auto size = info.at("size_bytes").get<std::uint64_t>();
  const auto path = input(name);
  require(fs::file_size(path) == size, "input metadata byte count differs: " + name);
  return {id,path,size,digest,media,source};
}
fs::path Context::asset(const std::string& name) const {
  require(impl_->assets.contains(name) && impl_->assets.at(name).is_string(),
          "unknown configured asset: " + name);
  return regular_path(impl_->assets.at(name).get<std::string>());
}
fs::path Context::output_path(const std::string& relative) const {
  fs::path path(relative);
  require(!path.empty() && !path.has_root_path() && relative.front() != '\\' &&
          (relative.size() < 2 || relative[1] != ':'),
          "output path must be relative and have no root or drive prefix");
  for (const auto& part : path)
    require(part != ".." && part != ".", "output path must stay inside attempt directory");
  const auto full = impl_->root / path;
  no_symlink(full);
  fs::create_directories(full.parent_path());
  return full;
}
void Context::register_output(const std::string& relative, const std::string& name,
                              const std::string& media_type, bool primary) {
  check_cancelled();
  require(!name.empty() && name.size() <= 128 && name != "." && name != ".." &&
          std::ranges::all_of(name, [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
          }),
          "invalid output artifact name");
  require(!media_type.empty() && media_type.size() <= 128 &&
          media_type.find_first_of("\r\n") == std::string::npos,
          "invalid output media type");
  const auto path = output_path(relative);
  require(fs::is_regular_file(path), "registered output must be a regular file");
  for (const auto& artifact : impl_->artifacts) {
    require(artifact.at("name") != name && artifact.at("path") != relative,
            "duplicate output artifact");
    require(!primary || !artifact.at("primary").get<bool>(),
            "only one output artifact may be primary");
  }
  impl_->artifacts.push_back({{"name",name},{"path",relative},{"media_type",media_type},{"primary",primary}});
}

class WorkflowHost::Impl {
 public:
  explicit Impl(const std::string& text) : owner(std::this_thread::get_id()) {
    const auto config = Json::parse(text);
    require(config.is_object() && (config.at("schema_version") == 1 || config.at("schema_version") == 2),
            "unsupported workflow host schema_version");
    schema_version = config.at("schema_version").get<int>();
    if (schema_version == 2) {
      require(config.at("schema_version").is_number_integer(), "workflow host schema_version must be an integer");
      const std::set<std::string> keys{"schema_version","adapter","models","assets","limits","tensor_adapter","custom"};
      for (const auto& [key,value] : config.items())
        require(keys.contains(key), "unknown workflow host configuration field: " + key);
    }
    adapter = config.at("adapter").get<std::string>();
    require(adapter == "tensor.v1" || adapter == "custom.v1", "unknown compiled workflow adapter: " + adapter);
    const auto limits = config.value("limits", Json::object());
    const auto json_limit = limits.value("max_json_bytes", 1048576LL);
    const auto tensor_limit = limits.value("max_tensor_bytes", 67108864LL);
    require(json_limit > 0 && json_limit <= 67108864 && tensor_limit > 0 && tensor_limit <= 268435456,
            "workflow byte limits must be positive and bounded");
    maximum_json = static_cast<std::size_t>(json_limit);
    models.maximum = static_cast<std::size_t>(tensor_limit);
    require(text.size() <= maximum_json, "configuration exceeds max_json_bytes");
    if (schema_version == 2) {
      const auto custom = config.value("custom", Json::object());
      require(custom.is_object(), "custom configuration must be a JSON object");
      require(adapter == "custom.v1" || custom.empty(), "custom configuration requires custom.v1");
      configuration = custom.dump();
      require(configuration.size() <= maximum_json, "custom configuration exceeds max_json_bytes");
    }
    assets = config.value("assets", Json::object());
    require(assets.is_object(), "assets must be a fixed name-to-path object");
    for (const auto& [name,path] : assets.items()) {
      require(identifier(name), "invalid asset name");
      (void)regular_path(path.get<std::string>());
    }
#ifdef PNMS_HAS_MOCK
    engine.register_backend(sdk::create_mock_backend());
#endif
#ifdef PNMS_HAS_AOTI
    engine.register_backend(sdk::create_aoti_backend());
#endif
#ifdef PNMS_HAS_TENSORRT
    engine.register_backend(sdk::create_tensorrt_backend());
#endif
#ifdef PNMS_HAS_ONNXRUNTIME
    engine.register_backend(sdk::create_onnxruntime_backend());
#endif
    const auto& definitions = config.at("models");
    require(definitions.is_object() && !definitions.empty() && definitions.size() <= 16,
            "models must contain one to sixteen fixed aliases");
    for (const auto& [alias,value] : definitions.items()) {
      require(identifier(alias), "invalid model alias");
      const auto& fixture = value.at("self_test");
      require(fixture.is_object() && fixture.contains("inputs") && fixture.contains("expected_outputs"),
              "each model requires an authored known-input self_test fixture");
      sdk::v1::ExecutorOptions options;
      options.backend = value.at("backend").get<std::string>();
      require(!options.backend.empty(), "an explicit backend is required; fallback is disabled");
      options.device = {sdk::parse_device_type(value.at("device").get<std::string>()),
                        value.value("device_index",0)};
      require(options.device.index >= 0, "device_index must not be negative");
      const auto path = value.at("package_path").get<std::string>();
      auto model = engine.load_model(path);
      for (const auto& spec : model.manifest().inputs) (void)static_count(spec,models.maximum);
      for (const auto& spec : model.manifest().outputs) (void)static_count(spec,models.maximum);
      exact_names(fixture.at("inputs"),model.manifest().inputs,"self_test inputs");
      exact_names(fixture.at("expected_outputs"),model.manifest().outputs,"self_test expected_outputs");
      auto executor = model.create_executor(options);
      require(executor.capabilities().accepts_cpu_inputs && executor.capabilities().caller_owned_cpu_outputs,
              "workflow backend must support CPU inputs and owned CPU outputs");
      models.entries.emplace(alias,ResidentModel{std::move(model),std::move(executor),fixture,path});
    }
    if (adapter == "tensor.v1") {
      mapping = config.at("tensor_adapter");
      model_alias = mapping.at("model_alias").get<std::string>();
      const auto& manifest = models.get(model_alias).model.manifest();
      exact_names(mapping.at("inputs"),manifest.inputs,"tensor_adapter inputs");
      exact_names(mapping.at("outputs"),manifest.outputs,"tensor_adapter outputs");
      for (const auto& direction : {"inputs","outputs"}) {
        std::set<std::string> fields;
        for (const auto& [name,entry] : mapping.at(direction).items()) {
          const auto field = entry.at(std::string(direction) == "inputs" ? "request_field" : "response_field").get<std::string>();
          require(identifier(field) && fields.insert(field).second,"mapping fields must be unique identifiers");
          const auto codec = entry.at("codec").get<std::string>();
          require(codec == "json-array-fp32" || codec == "raw-fp32-le","unsupported tensor codec: " + codec);
          if (codec == "raw-fp32-le")
            require(std::endian::native == std::endian::little,"raw-fp32-le requires a little-endian host");
          if (codec == "raw-fp32-le" && std::string(direction) == "outputs")
            require(identifier(name) && name.size() <= 124,
                    "raw output tensor name must be a safe identifier of at most 124 characters");
        }
      }
    } else {
#ifdef PNMS_HAS_CUSTOM_WORKFLOW
      workflow = create_custom_workflow();
      require(workflow != nullptr,"custom workflow factory returned null");
#else
      throw std::invalid_argument("custom.v1 requires a source-linked custom workflow factory");
#endif
    }
  }
  void check_thread() const {
    require(owner == std::this_thread::get_id(),"workflow host must stay on its owning thread");
  }
  std::string self_test(const std::function<bool()>& cancellation) {
    check_thread();
    require(!startup_failed,"workflow startup previously failed; create a new host");
    if (ready) return startup_report;
    try {
      return prepare_startup(cancellation);
    } catch (...) {
      startup_failed = true;
      throw;
    }
  }
  std::string prepare_startup(const std::function<bool()>& cancellation) {
    cancelled(cancellation);
    Json reports = Json::array();
    for (const auto& [alias,resident] : models.entries) {
      cancelled(cancellation);
      const auto& manifest = resident.model.manifest();
      std::vector<sdk::OwnedTensor> owned;
      std::vector<sdk::TensorView> views;
      for (const auto& input : manifest.inputs)
        owned.emplace_back(input.name,input.dtype,sdk::Device{},input.shape,
          bytes_of(json_values(resident.fixture.at("inputs").at(input.name),input,models.maximum)));
      for (const auto& tensor : owned) views.push_back(tensor.view());
      const auto result = models.infer(alias,views);
      const double rtol = resident.fixture.value("rtol",1e-5);
      const double atol = resident.fixture.value("atol",1e-6);
      require(std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) && atol >= 0,
              "self_test tolerances must be finite and nonnegative");
      Json outputs = Json::array();
      for (const auto& output : manifest.outputs) {
        const auto actual = output_values(result.output(output.name),output,models.maximum);
        const auto expected = json_values(resident.fixture.at("expected_outputs").at(output.name),output,models.maximum);
        const auto compared = sdk::v1::compare_self_test_float32(actual,expected,rtol,atol);
        outputs.push_back({{"name",output.name},{"dtype","float32"},{"shape",output.shape},
                           {"max_absolute_error",compared.max_absolute_error},
                           {"max_relative_error",compared.max_relative_error}});
      }
      cancelled(cancellation);
      reports.push_back({{"alias",alias},{"package",resident.package_path},
                         {"backend",resident.executor.backend_name()},{"outputs",outputs}});
    }
    if (workflow) {
      InitializationContext::Impl state{&models,&assets,&configuration,&cancellation};
      InitializationContext context(&state);
      context.check_cancelled();
      workflow->initialize(context);
      context.check_cancelled();
    }
    Json report{{"status","passed"},{"policy","expected-fp32-v1"},
      {"session_load_count",models.entries.size()},{"models",reports}};
    if (schema_version == 2) {
      report["host_schema_version"] = 2;
      report["initialization"] = {{"status",workflow ? "passed" : "not_required"},{"calls",workflow ? 1 : 0}};
    }
    startup_report = report.dump();
    require(startup_report.size() <= maximum_json,"startup report exceeds max_json_bytes");
    ready = true;
    return startup_report;
  }
  std::string run(const std::string& text,std::function<bool()> cancellation) {
    check_thread();
    require(ready,"workflow startup self_test must pass before requests");
    cancelled(cancellation);
    require(text.size() <= maximum_json,"request exceeds max_json_bytes");
    const auto request = Json::parse(text);
    require(request.is_object() && request.at("parameters").is_object(),"request parameters must be an object");
    const auto request_id = request.at("request_id").get<std::string>();
    require(!request_id.empty() && request_id.size() <= 256,"invalid request_id");
    fs::path root(request.at("output_dir").get<std::string>());
    require(root.is_absolute(),"output_dir must be an admitted absolute attempt directory");
    no_symlink(root);
    fs::create_directories(root);
    Context::Impl state{&models,request.value("inputs",Json::object()),assets,
                        fs::canonical(root),Json::array(),std::move(cancellation),&configuration,
                        request.value("input_metadata",Json::object())};
    require(state.inputs.is_object(),"admitted inputs must be an object");
    require(state.input_metadata.is_object(),"input_metadata must be an object");
    for (const auto& [name,info] : state.input_metadata.items())
      require(state.inputs.contains(name), "input metadata refers to unknown admitted input: " + name);
    Context context(&state);
    Outcome outcome;
    if (workflow) outcome = workflow->run({request_id,request.at("parameters").dump()},context);
    else outcome = tensor(request.at("parameters"),context);
    context.check_cancelled();
    const auto payload = Json::parse(outcome.payload_json);
    auto metadata = Json::parse(outcome.metadata_json);
    require(payload.is_object() && metadata.is_object(),"workflow payload/metadata must be JSON objects");
    metadata["adapter"] = adapter;
    metadata["session_load_count"] = models.entries.size();
    const auto result = Json{{"payload",payload},{"artifacts",state.artifacts},{"metadata",metadata}}.dump();
    require(result.size() <= maximum_json,"result exceeds max_json_bytes; use raw-fp32-le outputs");
    return result;
  }
 private:
  Outcome tensor(const Json& parameters,Context& context) {
    const auto& manifest = context.model(model_alias);
    require(parameters.size() == manifest.inputs.size(),"request fields must match tensor mapping exactly");
    std::vector<sdk::OwnedTensor> owned;
    std::vector<sdk::TensorView> views;
    for (const auto& spec : manifest.inputs) {
      const auto& entry = mapping.at("inputs").at(spec.name);
      const auto& value = parameters.at(entry.at("request_field").get<std::string>());
      std::vector<std::byte> bytes;
      if (entry.at("codec") == "json-array-fp32")
        bytes = bytes_of(json_values(value,spec,models.maximum));
      else {
        const auto count = static_count(spec,models.maximum);
        require(value.is_object() && value.size() == 6 && value.at("dtype") == "float32" &&
                value.at("shape") == Json(spec.shape) && value.at("byte_order") == "little" &&
                value.at("layout") == "contiguous" && value.at("size_bytes") == count*sizeof(float),
                "raw tensor descriptor differs from declared static FP32 contract");
        const auto path = context.input(value.at("artifact").get<std::string>());
        require(fs::file_size(path) == count*sizeof(float),"raw tensor file byte count differs");
        bytes.resize(count*sizeof(float));
        std::ifstream stream(path,std::ios::binary);
        require(bool(stream.read(reinterpret_cast<char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()))),"cannot read admitted raw tensor");
      }
      owned.emplace_back(spec.name,spec.dtype,sdk::Device{},spec.shape,std::move(bytes));
    }
    for (const auto& tensor : owned) views.push_back(tensor.view());
    // Check every destination before inference or the first write, so a later
    // conflict cannot leave earlier raw tensors behind in this attempt.
    for (const auto& spec : manifest.outputs) {
      if (mapping.at("outputs").at(spec.name).at("codec") == "raw-fp32-le") {
        const auto path = context.output_path(spec.name + ".f32");
        require(!fs::exists(path),"raw output already exists in attempt directory");
      }
    }
    const auto result = context.infer(model_alias,views);
    Json payload = Json::object();
    bool primary_output = true;
    for (const auto& spec : manifest.outputs) {
      const auto& entry = mapping.at("outputs").at(spec.name);
      const auto field = entry.at("response_field").get<std::string>();
      const auto values = output_values(result.output(spec.name),spec,models.maximum);
      if (entry.at("codec") == "json-array-fp32") payload[field] = values;
      else {
        const auto name = spec.name + ".f32";
        const auto path = context.output_path(name);
        require(!fs::exists(path),"raw output already exists in attempt directory");
        std::ofstream stream(path,std::ios::binary);
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size()*sizeof(float)));
        stream.close();
        require(bool(stream),"cannot write raw output tensor");
        context.register_output(name,name,"application/octet-stream",primary_output);
        primary_output = false;
        payload[field] = {{"artifact",name},{"dtype","float32"},{"shape",spec.shape},
                          {"byte_order","little"},{"layout","contiguous"},{"size_bytes",values.size()*sizeof(float)}};
      }
    }
    return {payload.dump(),"{}"};
  }
  sdk::v1::Engine engine;
  Models models;
  std::string adapter,model_alias,startup_report,configuration{"{}"};
  Json mapping,assets;
  std::size_t maximum_json{1048576};
  int schema_version{1};
  std::thread::id owner;
  bool ready{false},startup_failed{false};
  std::unique_ptr<Workflow> workflow;
};
WorkflowHost::WorkflowHost(const std::string& text) : impl_(std::make_unique<Impl>(text)) {}
WorkflowHost::~WorkflowHost() = default;
std::string WorkflowHost::self_test(std::function<bool()> callback) { return impl_->self_test(callback); }
std::string WorkflowHost::run(const std::string& text,std::function<bool()> callback) {
  return impl_->run(text,std::move(callback));
}
}  // namespace physicsnemo::workflow::v1
