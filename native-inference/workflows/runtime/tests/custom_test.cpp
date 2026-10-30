#include "physicsnemo/workflow/v1/host.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "custom/observations.hpp"

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F&& function, const std::string& text) {
  try { function(); }
  catch (const std::exception& error) {
    check(std::string(error.what()).find(text) != std::string::npos,
          "unexpected rejection: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection: " + text);
}
}
int main() {
  using Json = nlohmann::json;
  using physicsnemo::workflow::v1::WorkflowHost;
  namespace observed = physicsnemo::workflow::v1::test;
  const auto root = std::filesystem::canonical(std::filesystem::temp_directory_path()) /
      "physicsnemo-custom-workflow-fixture";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "package");
  try {
    std::ofstream(root / "package/model.json") << R"({"format_version":1,"model":{"name":"identity","version":"1"},"inputs":[{"name":"input","dtype":"float32","shape":[4]}],"outputs":[{"name":"output","dtype":"float32","shape":[4]}],"artifacts":[{"backend":"mock","target":"cpu","precision":"fp32","path":"identity.mock"}]})";
    std::ofstream(root / "package/identity.mock") << "identity";
    const Json values = {0.0,1.0,-1.0,4.0};
    Json config{{"schema_version",1},{"adapter","custom.v1"},
      {"models",{{"primary",{{"package_path",(root / "package").string()},
        {"backend","mock"},{"device","cpu"},
        {"self_test",{{"inputs",{{"input",values}}},{"expected_outputs",{{"output",values}}}}}}}}}};
    physicsnemo::workflow::v1::WorkflowHost host(config.dump());
    host.self_test();
    Json request{{"request_id","custom-case"},{"parameters",{{"values",values}}},
      {"output_dir",(root / "outputs").string()},{"inputs",Json::object()}};
    const auto result = Json::parse(host.run(request.dump()));
    if (result.at("payload").at("values") != Json::array({10.0,11.0,9.0,14.0}))
      throw std::runtime_error("source-linked custom workflow did not infer and postprocess");
    if (result.at("artifacts").at(0).at("path") != "custom.f32")
      throw std::runtime_error("custom output registration failed");
    for (const auto& mode : {"escape","alias","artifact-space","artifact-long","media-newline"}) {
      request["parameters"]["mode"] = mode;
      bool rejected = false;
      try { host.run(request.dump()); } catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("unsafe custom context operation accepted");
    }
    request["parameters"].erase("mode");
    request["output_dir"] = (root / "cancelled").string();
    int cancellation_checks = 0;
    bool cancelled = false;
    try {
      host.run(request.dump(), [&] { return ++cancellation_checks >= 3; });
    } catch (const std::runtime_error& error) {
      cancelled = std::string(error.what()).find("cancelled") != std::string::npos;
    }
    if (!cancelled || cancellation_checks != 3 ||
        std::filesystem::exists(root / "cancelled/custom.f32"))
      throw std::runtime_error("post-inference cancellation did not prevent custom output publication");

    auto current_config = config;
    current_config["schema_version"] = 2;
    current_config["custom"] = {{"offset",2.0},{"multiplier",3.0}};
    WorkflowHost configured(current_config.dump());
    const auto configured_startup = Json::parse(configured.self_test());
    check(configured_startup.at("initialization") == Json{{"status","passed"},{"calls",1}},
          "schema2 startup did not record one successful initialization");
    const auto initializations = observed::initializations();
    current_config["custom"]["offset"] = 999;
    request["output_dir"] = (root / "configured").string();
    const auto configured_result = Json::parse(configured.run(request.dump()));
    check(configured_result.at("payload").at("values") == Json::array({2.0,5.0,-1.0,14.0}),
          "two immutable settings did not control custom computation");
    check(configured_result.at("metadata").at("configuration") == Json{{"offset",2.0},{"multiplier",3.0}},
          "request context did not expose the same immutable initialization configuration");
    configured.run(request.dump());
    check(Json::parse(configured.self_test()) == configured_startup &&
          observed::initializations() == initializations,
          "repeated calls reinitialized a ready custom workflow");

    auto absent_config = config;
    absent_config["schema_version"] = 2;
    WorkflowHost defaults(absent_config.dump());
    defaults.self_test();
    check(Json::parse(defaults.run(request.dump())).at("metadata").at("configuration") == Json::object(),
          "absent custom configuration did not default to an empty object");
    auto legacy_config = config;
    legacy_config["custom"] = {{"offset",999}};
    WorkflowHost legacy(legacy_config.dump());
    const auto legacy_report = Json::parse(legacy.self_test());
    check(!legacy_report.contains("initialization") &&
          Json::parse(legacy.run(request.dump())).at("payload").at("values") == Json::array({10.0,11.0,9.0,14.0}),
          "schema1 custom configuration or startup report changed legacy behavior");

    const auto asset = root / "offset.txt";
    std::ofstream(asset) << "7";
    auto asset_config = absent_config;
    asset_config["assets"] = {{"offset",asset.string()}};
    asset_config["custom"] = {{"offset_asset","offset"}};
    WorkflowHost initialized_asset(asset_config.dump());
    initialized_asset.self_test();
    std::filesystem::remove(asset);
    check(Json::parse(initialized_asset.run(request.dump())).at("payload").at("values") == Json::array({7.0,8.0,6.0,11.0}),
          "initialization did not prepare owned state from a packaged asset");

    auto bad = absent_config;
    bad["custom"] = {{"fail_initialize",true}};
    WorkflowHost failed(bad.dump());
    const auto before_failure = observed::initializations();
    rejects([&] { failed.self_test(); }, "authored initialization failure");
    rejects([&] { failed.self_test(); }, "startup previously failed");
    rejects([&] { failed.run(request.dump()); }, "self_test must pass");
    check(observed::initializations() == before_failure + 1,
          "failed initialization was retried on partially initialized state");

    WorkflowHost cancelled_initialization(absent_config.dump());
    const auto before_cancellation = observed::initializations();
    rejects([&] {
      cancelled_initialization.self_test([&] { return observed::initializations() > before_cancellation; });
    }, "cancelled");
    rejects([&] { cancelled_initialization.self_test(); }, "startup previously failed");
    check(observed::initializations() == before_cancellation + 1,
          "cancelled initialization was retried");

    bad = absent_config;
    bad["models"]["primary"]["self_test"]["expected_outputs"]["output"][0] = 99;
    WorkflowHost failed_model(bad.dump());
    const auto before_model_check = observed::initializations();
    rejects([&] { failed_model.self_test(); }, "self_test output mismatch");
    check(observed::initializations() == before_model_check,
          "custom initialization ran before authored model checks passed");
    rejects([&] { failed_model.self_test(); }, "startup previously failed");

    for (const auto& invalid : {Json::array(),Json(nullptr),Json("settings")}) {
      bad = absent_config;
      bad["custom"] = invalid;
      rejects([&] { WorkflowHost ignored(bad.dump()); }, "custom configuration must be a JSON object");
    }
    bad = absent_config;
    bad["unknown_setting"] = true;
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "unknown workflow host configuration field");
    bad = absent_config;
    bad["limits"]["max_json_bytes"] = 1024;
    bad["custom"]["oversized"] = std::string(1024,'x');
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "max_json_bytes");

    const auto before_destruction = observed::destructions();
    std::string owner_error;
    std::thread owner([&] {
      try {
        WorkflowHost owned(absent_config.dump());
        owned.self_test();
        owned.run(request.dump());
      } catch (const std::exception& error) { owner_error = error.what(); }
    });
    owner.join();
    check(owner_error.empty() && observed::destructions() == before_destruction + 1 && observed::wrong_threads() == 0,
          "custom construction/initialization/destruction escaped the owner thread: " + owner_error);

    const auto admitted = root / "admitted.bin";
    std::ofstream(admitted) << "data";
    request["inputs"] = {{"features",admitted.string()}};
    request["input_metadata"] = {{"features",{{"artifact_id","upload-123"},{"size_bytes",4},
      {"sha256",std::string(64,'a')},{"media_type","application/octet-stream"},{"source","request"}}}};
    request["parameters"]["mode"] = "input-info";
    const auto admitted_result = Json::parse(defaults.run(request.dump()));
    auto expected_info = request["input_metadata"]["features"];
    expected_info["path"] = admitted.string();
    check(admitted_result.at("metadata").at("input_info") == expected_info,
          "verified request identity did not reach the typed input accessor");
    request["input_metadata"]["features"]["source"] = "release";
    check(Json::parse(defaults.run(request.dump())).at("metadata").at("input_info").at("source") == "release",
          "fixed release input identity did not reach the typed accessor");
    request["input_metadata"]["features"]["size_bytes"] = 5;
    rejects([&] { defaults.run(request.dump()); }, "input metadata byte count differs");
    request["input_metadata"]["features"]["size_bytes"] = 4;
    request["input_metadata"]["features"]["sha256"] = "not-a-digest";
    rejects([&] { defaults.run(request.dump()); }, "input metadata sha256");
    request["input_metadata"]["features"]["sha256"] = std::string(64,'a');
    request["input_metadata"]["features"]["path"] = "/tmp/unchecked";
    rejects([&] { defaults.run(request.dump()); }, "input metadata fields");
    request["input_metadata"]["features"].erase("path");
    request.erase("input_metadata");
    rejects([&] { defaults.run(request.dump()); }, "verified input metadata is unavailable");
    std::filesystem::remove_all(root);
    std::cout << "custom source-linked workflow passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
