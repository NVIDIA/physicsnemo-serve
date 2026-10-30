#include "physicsnemo/workflow/v1/authoring.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <atomic>
#include <thread>
#include "observations.hpp"
namespace physicsnemo::workflow::v1 {
namespace {
std::atomic<int> initialization_count{0}, destruction_count{0}, wrong_thread_count{0};
class Custom final : public Workflow {
 public:
  ~Custom() override {
    ++destruction_count;
    if (owner != std::this_thread::get_id()) ++wrong_thread_count;
  }
  void initialize(InitializationContext& context) override {
    ++initialization_count;
    ++initialized;
    if (owner != std::this_thread::get_id()) ++wrong_thread_count;
    context.check_cancelled();
    configuration = context.configuration_json();
    const auto settings = nlohmann::json::parse(configuration);
    offset = settings.value("offset",10.0F);
    multiplier = settings.value("multiplier",1.0F);
    if (settings.contains("offset_asset")) {
      std::ifstream file(context.asset(settings.at("offset_asset").get<std::string>()));
      file >> offset;
      if (!file) throw std::runtime_error("cannot read initialization asset");
    }
    if (context.model("primary").inputs.at(0).shape != std::vector<std::int64_t>{4})
      throw std::runtime_error("initialization received wrong model manifest");
    if (settings.value("fail_initialize",false)) throw std::runtime_error("authored initialization failure");
  }
  Outcome run(const Request& request, Context& context) override {
    if (initialized != 1) throw std::runtime_error("custom workflow initialization did not run exactly once");
    if (configuration != context.configuration_json()) throw std::runtime_error("custom configuration changed between calls");
    const auto parameters = nlohmann::json::parse(request.parameters_json);
    const auto mode = parameters.value("mode", "");
    auto metadata = nlohmann::json{{"custom",true},{"initializations",initialized},
      {"configuration",nlohmann::json::parse(configuration)}};
    if (mode == "input-info") {
      const auto input = context.input_info(parameters.value("input_name","features"));
      if (input.path != context.input(parameters.value("input_name","features")))
        throw std::runtime_error("input metadata replaced the admitted path");
      metadata["input_info"] = {{"artifact_id",input.artifact_id},{"path",input.path.string()},
        {"size_bytes",input.size_bytes},{"sha256",input.sha256},{"media_type",input.media_type},{"source",input.source}};
    }
    if (mode == "escape") context.output_path("../escape");
    const auto values = parameters.at("values").get<std::vector<float>>();
    inference::TensorView input{"input", inference::DType::kFloat32,
        {inference::DeviceType::kCpu,0}, {4}, values.data(), values.size()*sizeof(float)};
    const auto result = context.infer(parameters.value("mode", "") == "alias" ? "other" : "primary", {input});
    const auto output = result.output("output");
    const auto* data = static_cast<const float*>(output.data);
    std::vector<float> predictions(data,data+output.byte_size/sizeof(float));
    for (auto& value : predictions) value = value * multiplier + offset;
    std::ofstream file(context.output_path("custom.f32"),std::ios::binary);
    file.write(reinterpret_cast<const char*>(predictions.data()),predictions.size()*sizeof(float));
    file.close();
    const auto artifact_name = mode == "artifact-space" ? "custom field.f32" :
        mode == "artifact-long" ? std::string(129,'a') : "custom.f32";
    const auto media_type = mode == "media-newline" ? "application/octet-stream\r\nX-Test: invalid" :
        "application/octet-stream";
    context.register_output("custom.f32",artifact_name,media_type,true);
    return {nlohmann::json{{"values",predictions}}.dump(),metadata.dump()};
  }
 private:
  std::thread::id owner{std::this_thread::get_id()};
  int initialized{0};
  float offset{10.0F},multiplier{1.0F};
  std::string configuration;
};
}
namespace test {
int initializations() { return initialization_count.load(); }
int destructions() { return destruction_count.load(); }
int wrong_threads() { return wrong_thread_count.load(); }
}
std::unique_ptr<Workflow> create_custom_workflow() { return std::make_unique<Custom>(); }
}
