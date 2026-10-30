#include "physicsnemo/workflow/v1/host.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
using Json = nlohmann::json;
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
  using physicsnemo::workflow::v1::WorkflowHost;
  auto root = std::filesystem::canonical(std::filesystem::temp_directory_path()) /
              "physicsnemo-workflow-host-contract";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "package");
  try {
    Json manifest{{"format_version",1},{"model",{{"name","identity"},{"version","1"}}},
      {"inputs",Json::array({{{"name","input"},{"dtype","float32"},{"shape",{4}}}})},
      {"outputs",Json::array({{{"name","output"},{"dtype","float32"},{"shape",{4}}}})},
      {"artifacts",Json::array({{{"backend","mock"},{"target","cpu"},{"precision","fp32"},{"path","identity.mock"}}})}};
    std::ofstream(root / "package/model.json") << manifest.dump();
    std::ofstream(root / "package/identity.mock") << "identity";
    const Json values = {0.0,1.0,-1.0,4.0};
    Json config{{"schema_version",1},{"adapter","tensor.v1"},
      {"models",{{"primary",{{"package_path",(root / "package").string()},
         {"backend","mock"},{"device","cpu"},{"device_index",0},
         {"self_test",{{"inputs",{{"input",values}}},{"expected_outputs",{{"output",values}}},{"rtol",0.0},{"atol",0.0}}}}}}},
      {"tensor_adapter",{{"model_alias","primary"},
        {"inputs",{{"input",{{"request_field","values"},{"codec","json-array-fp32"}}}}},
        {"outputs",{{"output",{{"response_field","values"},{"codec","json-array-fp32"}}}}}}}};
    WorkflowHost host(config.dump());
    auto schema_two = config;
    schema_two["schema_version"] = 2;
    try {
      WorkflowHost current(schema_two.dump());
      check(Json::parse(current.self_test()).at("status") == "passed",
            "schema2 host did not pass its authored model checks");
    } catch (const std::exception& error) {
      throw std::runtime_error("schema2 lifecycle configuration must be accepted: " + std::string(error.what()));
    }
    auto unknown_schema = config;
    unknown_schema["schema_version"] = 3;
    rejects([&] { WorkflowHost ignored(unknown_schema.dump()); }, "unsupported workflow host schema_version");
    Json request{{"request_id","case-1"},{"parameters",{{"values",values}}},
                 {"output_dir",(root / "outputs").string()},{"inputs",Json::object()}};
    rejects([&] { host.run(request.dump()); }, "self_test must pass");
    auto startup = Json::parse(host.self_test());
    check(startup.at("status") == "passed", "known-input startup self-test must execute and pass");
    check(startup.at("models").at(0).at("outputs").at(0).at("max_absolute_error") == 0,
          "startup did not compare expected output values");
    auto result = Json::parse(host.run(request.dump()));
    check(result.at("payload").at("values") == values, "tensor adapter changed identity output");
    check(result.at("metadata").at("session_load_count") == 1, "wrong session count");
    std::filesystem::remove(root / "package/model.json");
    check(Json::parse(host.run(request.dump())).at("payload").at("values") == values,
          "resident executor reloaded package per request");
    check(Json::parse(host.self_test()) == startup, "startup report was not retained");
    std::ofstream(root / "package/model.json") << manifest.dump();
    rejects([&] { host.run(request.dump(), [] { return true; }); }, "cancelled");
    request["parameters"]["values"] = {1.0};
    rejects([&] { host.run(request.dump()); }, "element count");
    request["parameters"]["values"] = {1.0,2.0,true,4.0};
    rejects([&] { host.run(request.dump()); }, "numeric");
    request["parameters"]["values"] = values;
    request["parameters"]["extra"] = 1;
    rejects([&] { host.run(request.dump()); }, "mapping exactly");
    request["parameters"].erase("extra");
    std::string foreign_thread_error;
    std::thread foreign([&] {
      try { host.self_test(); }
      catch (const std::exception& error) { foreign_thread_error = error.what(); }
    });
    foreign.join();
    check(foreign_thread_error.find("owning thread") != std::string::npos,
          "host accepted another OS thread");

    auto bad = config;
    bad["models"]["primary"]["self_test"]["expected_outputs"]["output"][2] = 17;
    WorkflowHost mismatched(bad.dump());
    rejects([&] { mismatched.self_test(); }, "self_test output mismatch");
    rejects([&] { mismatched.run(request.dump()); }, "self_test must pass");
    bad = config;
    bad["models"]["primary"]["self_test"].erase("expected_outputs");
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "authored known-input");
    bad = config;
    bad["tensor_adapter"]["model_alias"] = "other";
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "unknown fixed model alias");
    bad = config;
    bad["models"]["primary"]["backend"] = "";
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "explicit backend");
    bad = config;
    bad["limits"]["max_tensor_bytes"] = 8;
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "max_tensor_bytes");
    bad = config;
    bad["tensor_adapter"]["inputs"]["input"]["codec"] = "python-pickle";
    rejects([&] { WorkflowHost ignored(bad.dump()); }, "unsupported tensor codec");
    bad = config;
    bad["limits"]["max_json_bytes"] = 1024;
    WorkflowHost bounded(bad.dump());
    bounded.self_test();
    rejects([&] { bounded.run(request.dump() + std::string(1024,' ')); }, "max_json_bytes");

    auto raw_config = config;
    raw_config["tensor_adapter"]["inputs"]["input"]["codec"] = "raw-fp32-le";
    raw_config["tensor_adapter"]["outputs"]["output"]["codec"] = "raw-fp32-le";
    WorkflowHost raw(raw_config.dump());
    raw.self_test();
    const auto binary = root / "input.f32";
    const std::vector<float> floats{0,1,-1,4};
    const auto write_raw = [&](const std::vector<float>& numbers) {
      std::ofstream out(binary,std::ios::binary);
      out.write(reinterpret_cast<const char*>(numbers.data()),numbers.size()*sizeof(float));
    };
    write_raw(floats);
    request["parameters"]["values"] = {{"artifact","features"},{"dtype","float32"},
      {"shape",{4}},{"byte_order","little"},{"layout","contiguous"},{"size_bytes",16}};
    request["inputs"]["features"] = binary.string();
    request["output_dir"] = (root / "raw-output").string();
    auto raw_result = Json::parse(raw.run(request.dump()));
    check(raw_result.at("artifacts").size() == 1 && raw_result.at("artifacts").at(0).at("path") == "output.f32",
          "raw output was not registered as a relative artifact");
    check(raw_result.at("payload").at("values").at("shape") == Json::array({4}),
          "raw output lost the model tensor shape");
    std::ifstream produced(root / "raw-output/output.f32",std::ios::binary);
    std::vector<float> roundtrip(4);
    produced.read(reinterpret_cast<char*>(roundtrip.data()),16);
    produced.close();
    check(roundtrip == floats, "raw tensor round trip changed values");
    rejects([&] { raw.run(request.dump()); }, "already exists");
    request["output_dir"] = (root / "raw-output-2").string();
    request["parameters"]["values"]["artifact"] = "unadmitted";
    rejects([&] { raw.run(request.dump()); }, "unknown admitted input");
    request["parameters"]["values"]["artifact"] = "features";
    request["parameters"]["values"]["path"] = binary.string();
    rejects([&] { raw.run(request.dump()); }, "raw tensor descriptor");
    request["parameters"]["values"].erase("path");
    write_raw({1,2});
    rejects([&] { raw.run(request.dump()); }, "file byte count");
    write_raw({1,2,std::numeric_limits<float>::quiet_NaN(),4});
    rejects([&] { raw.run(request.dump()); }, "finite");
    write_raw(floats);
    std::filesystem::create_symlink(binary,root / "linked.f32");
    request["inputs"]["features"] = (root / "linked.f32").string();
    rejects([&] { raw.run(request.dump()); }, "symlink");
    request["inputs"]["features"] = binary.string();
    std::filesystem::create_directory_symlink(root,root / "linked-dir");
    request["output_dir"] = (root / "linked-dir/escape").string();
    rejects([&] { raw.run(request.dump()); }, "symlink");

    const std::string long_output_name(125,'a');
    manifest["outputs"][0]["name"] = long_output_name;
    std::ofstream(root / "package/model.json") << manifest.dump();
    auto oversized_raw_name = raw_config;
    oversized_raw_name["models"]["primary"]["self_test"]["expected_outputs"] = {{long_output_name,values}};
    oversized_raw_name["tensor_adapter"]["outputs"] = {
      {long_output_name,{{"response_field","values"},{"codec","raw-fp32-le"}}}};
    rejects([&] { WorkflowHost ignored(oversized_raw_name.dump()); }, "at most 124 characters");
    const std::string maximum_output_name(124,'a');
    manifest["outputs"][0]["name"] = maximum_output_name;
    std::ofstream(root / "package/model.json") << manifest.dump();
    auto maximum_raw_name = oversized_raw_name;
    maximum_raw_name["models"]["primary"]["self_test"]["expected_outputs"] = {{maximum_output_name,values}};
    maximum_raw_name["tensor_adapter"]["outputs"] = {
      {maximum_output_name,{{"response_field","values"},{"codec","raw-fp32-le"}}}};
    WorkflowHost longest(maximum_raw_name.dump());
    longest.self_test();
    request["output_dir"] = (root / "maximum-name-output").string();
    const auto maximum_result = Json::parse(longest.run(request.dump()));
    check(maximum_result.at("artifacts").at(0).at("name").get<std::string>().size() == 128,
          "maximum valid raw artifact name was not retained");
    manifest["outputs"][0]["name"] = "output";
    manifest["inputs"][0]["shape"] = {-1};
    std::ofstream(root / "package/model.json") << manifest.dump();
    rejects([&] { WorkflowHost ignored(config.dump()); }, "positive static");
    std::filesystem::remove_all(root);
    std::cout << "workflow host contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
