#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "physicsnemo/inference/backend.hpp"
#include "physicsnemo/inference/manifest.hpp"
#include "physicsnemo/inference/runtime.hpp"

#ifdef PNMIR_HAS_AOTI
#include "physicsnemo/inference/backends/aoti.hpp"
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
#include "physicsnemo/inference/backends/onnxruntime.hpp"
#endif
#ifdef PNMIR_HAS_TENSORRT
#include "physicsnemo/inference/backends/tensorrt.hpp"
#endif

namespace {

void usage() {
  std::cerr << "usage:\n"
            << "  physicsnemo-infer inspect <package>\n"
            << "  physicsnemo-infer run <package> --values 1,2,3 [--backend <name>]\n"
            << "  physicsnemo-infer run <package> --input-file <name>=<raw-file> ... "
               "[--input-shape <name>=<d0,d1,...>] ... "
               "[--output-file [<name>=]<raw-file>] ... [--backend <name>] "
               "[--device cpu|cuda[:index]] [--warmup N] "
               "[--iterations N] [--output-metadata <json-file>]\n";
}

struct FileInput {
  std::string name;
  std::filesystem::path path;
};

struct FileOutput {
  std::string name;
  std::filesystem::path path;
};

struct InputShape {
  std::string name;
  physicsnemo::inference::Shape shape;
};

std::size_t parse_count(const std::string& text, const std::string& option,
                        bool allow_zero);

FileInput parse_file_input(const std::string& text) {
  const auto separator = text.find('=');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == text.size()) {
    throw std::invalid_argument(
        "--input-file must use the form <name>=<raw-file>");
  }
  return {text.substr(0, separator), text.substr(separator + 1)};
}

FileOutput parse_file_output(const std::string& text) {
  const auto separator = text.find('=');
  if (separator == std::string::npos) {
    if (text.empty()) {
      throw std::invalid_argument("--output-file path cannot be empty");
    }
    return {"", text};
  }
  if (separator == 0 || separator + 1 == text.size()) {
    throw std::invalid_argument(
        "--output-file must use the form [<name>=]<raw-file>");
  }
  return {text.substr(0, separator), text.substr(separator + 1)};
}

InputShape parse_input_shape(const std::string& text) {
  const auto separator = text.find('=');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == text.size()) {
    throw std::invalid_argument(
        "--input-shape must use the form <name>=<d0,d1,...>");
  }
  InputShape result{text.substr(0, separator), {}};
  std::stringstream stream(text.substr(separator + 1));
  std::string item;
  while (std::getline(stream, item, ',')) {
    const auto dimension = parse_count(item, "--input-shape dimension", false);
    if (dimension >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("--input-shape dimension is too large");
    }
    result.shape.push_back(static_cast<std::int64_t>(dimension));
  }
  if (result.shape.empty()) {
    throw std::invalid_argument("--input-shape must contain at least one dimension");
  }
  return result;
}

void validate_actual_shape(const physicsnemo::inference::TensorSpec& spec,
                           const physicsnemo::inference::Shape& actual) {
  if (actual.size() != spec.shape.size()) {
    throw std::invalid_argument("rank mismatch for --input-shape: " + spec.name);
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] <= 0 ||
        (spec.shape[i] != -1 && actual[i] != spec.shape[i])) {
      throw std::invalid_argument("shape mismatch for --input-shape: " +
                                  spec.name);
    }
  }
}

physicsnemo::inference::Device parse_device(const std::string& text) {
  if (text == "cpu") return {};
  if (text == "cuda") return {physicsnemo::inference::DeviceType::kCuda, 0};
  constexpr std::string_view prefix{"cuda:"};
  if (!text.starts_with(prefix)) {
    throw std::invalid_argument("--device must be cpu, cuda, or cuda:<index>");
  }
  const std::string index_text = text.substr(prefix.size());
  std::size_t consumed = 0;
  const int index = std::stoi(index_text, &consumed);
  if (index < 0 || consumed != index_text.size()) {
    throw std::invalid_argument("invalid CUDA device index: " + index_text);
  }
  return {physicsnemo::inference::DeviceType::kCuda, index};
}

std::size_t parse_count(const std::string& text, const std::string& option,
                        bool allow_zero) {
  if (text.empty() || !std::ranges::all_of(text, [](unsigned char value) {
        return value >= '0' && value <= '9';
      })) {
    throw std::invalid_argument(option + " must be a non-negative integer");
  }
  const auto count = std::stoull(text);
  if (!allow_zero && count == 0) {
    throw std::invalid_argument(option + " must be greater than zero");
  }
  return static_cast<std::size_t>(count);
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(const std::vector<double>& sorted, std::size_t numerator) {
  const std::size_t rank = (numerator * sorted.size() + 99) / 100;
  return sorted[std::max<std::size_t>(rank, 1) - 1];
}

std::vector<float> parse_values(const std::string& text) {
  std::vector<float> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty input value");
    values.push_back(std::stof(item));
  }
  if (values.empty()) throw std::invalid_argument("at least one value is required");
  return values;
}

std::vector<std::byte> as_bytes(const std::vector<float>& values) {
  std::vector<std::byte> storage(values.size() * sizeof(float));
  std::memcpy(storage.data(), values.data(), storage.size());
  return storage;
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path,
                                  std::size_t expected_size) {
  const auto actual_size = std::filesystem::file_size(path);
  if (actual_size != expected_size) {
    throw std::invalid_argument("input file size does not match manifest: " +
                                path.string());
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open input file: " + path.string());
  }
  std::vector<std::byte> storage(expected_size);
  if (expected_size != 0 &&
      !input.read(reinterpret_cast<char*>(storage.data()),
                  static_cast<std::streamsize>(expected_size))) {
    throw std::runtime_error("cannot read input file: " + path.string());
  }
  return storage;
}

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::byte>& storage) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path.string());
  }
  if (!storage.empty()) {
    output.write(reinterpret_cast<const char*>(storage.data()),
                 static_cast<std::streamsize>(storage.size()));
  }
  if (!output) {
    throw std::runtime_error("cannot write output file: " + path.string());
  }
}

int inspect(const std::filesystem::path& path) {
  const auto package = physicsnemo::inference::ModelPackage::load(path);
  const auto& manifest = package.manifest();
  std::cout << manifest.model_name << " " << manifest.model_version << '\n';
  std::cout << "format: " << manifest.format_version << '\n';
  std::cout << "inputs:\n";
  for (const auto& input : manifest.inputs) {
    std::cout << "  " << input.name << " " << physicsnemo::inference::to_string(input.dtype)
              << " rank=" << input.shape.size() << '\n';
  }
  std::cout << "outputs:\n";
  for (const auto& output : manifest.outputs) {
    std::cout << "  " << output.name << " " << physicsnemo::inference::to_string(output.dtype)
              << " rank=" << output.shape.size() << '\n';
  }
  std::cout << "artifacts:\n";
  for (const auto& artifact : manifest.artifacts) {
    std::cout << "  " << artifact.backend << " "
              << physicsnemo::inference::to_string(artifact.target) << " "
              << artifact.precision << " " << artifact.path.string() << '\n';
    for (const auto& required : artifact.required_operators) {
      std::cout << "    requires " << required.id << "@" << required.abi
                << '\n';
    }
  }
  return 0;
}

int run(int argc, char** argv) {
  const std::filesystem::path package_path = argv[2];
  std::string values_text;
  std::vector<FileInput> file_inputs;
  std::vector<InputShape> input_shapes;
  std::vector<FileOutput> output_files;
  std::filesystem::path output_metadata;
  std::size_t warmup = 0;
  std::size_t iterations = 1;
  bool benchmark_requested = false;
  physicsnemo::inference::SessionOptions options;
  for (int i = 3; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--values" && i + 1 < argc) {
      values_text = argv[++i];
    } else if (argument == "--backend" && i + 1 < argc) {
      options.backend = argv[++i];
    } else if (argument == "--input-file" && i + 1 < argc) {
      file_inputs.push_back(parse_file_input(argv[++i]));
    } else if (argument == "--input-shape" && i + 1 < argc) {
      input_shapes.push_back(parse_input_shape(argv[++i]));
    } else if (argument == "--output-file" && i + 1 < argc) {
      output_files.push_back(parse_file_output(argv[++i]));
    } else if (argument == "--output-metadata" && i + 1 < argc) {
      output_metadata = argv[++i];
      if (output_metadata.empty()) {
        throw std::invalid_argument("--output-metadata path cannot be empty");
      }
    } else if (argument == "--device" && i + 1 < argc) {
      options.device = parse_device(argv[++i]);
    } else if (argument == "--warmup" && i + 1 < argc) {
      warmup = parse_count(argv[++i], "--warmup", true);
      benchmark_requested = true;
    } else if (argument == "--iterations" && i + 1 < argc) {
      iterations = parse_count(argv[++i], "--iterations", false);
      benchmark_requested = true;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + argument);
    }
  }
  if (values_text.empty() == file_inputs.empty()) {
    throw std::invalid_argument(
        "provide either --values or one or more --input-file arguments");
  }

  const auto package_load_start = std::chrono::steady_clock::now();
  const auto package = physicsnemo::inference::ModelPackage::load(package_path);
  const double package_load_ms =
      elapsed_ms(package_load_start, std::chrono::steady_clock::now());
  const auto& inputs = package.manifest().inputs;

  physicsnemo::inference::Runtime runtime;
  runtime.register_backend(physicsnemo::inference::create_mock_backend());
#ifdef PNMIR_HAS_AOTI
  physicsnemo::inference::register_aoti_exact_operators(runtime);
  runtime.register_backend(physicsnemo::inference::create_aoti_backend());
#endif
#ifdef PNMIR_HAS_ONNXRUNTIME
  runtime.register_backend(physicsnemo::inference::create_onnxruntime_backend());
#endif
#ifdef PNMIR_HAS_TENSORRT
  physicsnemo::inference::register_tensorrt_exact_operators(runtime);
  runtime.register_backend(physicsnemo::inference::create_tensorrt_backend());
#endif
  const auto session_create_start = std::chrono::steady_clock::now();
  auto session = runtime.create_session(package, options);
  const double session_create_ms =
      elapsed_ms(session_create_start, std::chrono::steady_clock::now());

  std::vector<physicsnemo::inference::OwnedTensor> owned_inputs;
  if (!values_text.empty()) {
    if (inputs.size() != 1 || inputs.front().dtype != physicsnemo::inference::DType::kFloat32) {
      throw std::invalid_argument(
          "--values supports one float32 input tensor package");
    }
    const auto values = parse_values(values_text);
    physicsnemo::inference::Shape actual_shape = inputs.front().shape;
    if (!input_shapes.empty()) {
      if (input_shapes.size() != 1 ||
          input_shapes.front().name != inputs.front().name) {
        throw std::invalid_argument(
            "--input-shape must name the --values input tensor");
      }
      actual_shape = std::move(input_shapes.front().shape);
      validate_actual_shape(inputs.front(), actual_shape);
    } else if (actual_shape.size() == 1 && actual_shape.front() == -1) {
      actual_shape.front() = static_cast<std::int64_t>(values.size());
    } else if (std::ranges::find(actual_shape, -1) != actual_shape.end()) {
      throw std::invalid_argument(
          "dynamic input requires --input-shape: " + inputs.front().name);
    }
    const std::size_t expected_values = physicsnemo::inference::element_count(actual_shape);
    if (values.size() != expected_values) {
      throw std::invalid_argument("--values count does not match input shape");
    }
    owned_inputs.emplace_back(inputs.front().name, physicsnemo::inference::DType::kFloat32,
                              physicsnemo::inference::Device{}, std::move(actual_shape),
                              as_bytes(values));
  } else {
    std::unordered_map<std::string, physicsnemo::inference::Shape> shapes_by_name;
    for (auto& input_shape : input_shapes) {
      if (!shapes_by_name
               .emplace(input_shape.name, std::move(input_shape.shape))
               .second) {
        throw std::invalid_argument("duplicate --input-shape: " +
                                    input_shape.name);
      }
    }
    owned_inputs.reserve(file_inputs.size());
    for (const auto& file_input : file_inputs) {
      const auto spec = std::find_if(
          inputs.begin(), inputs.end(), [&](const physicsnemo::inference::TensorSpec& candidate) {
            return candidate.name == file_input.name;
          });
      if (spec == inputs.end()) {
        throw std::invalid_argument("unknown input tensor: " + file_input.name);
      }
      const auto provided_shape = shapes_by_name.find(spec->name);
      physicsnemo::inference::Shape actual_shape =
          provided_shape == shapes_by_name.end() ? spec->shape
                                                 : provided_shape->second;
      if (std::ranges::find(actual_shape, -1) != actual_shape.end()) {
        throw std::invalid_argument(
            "dynamic input requires --input-shape: " + spec->name);
      }
      validate_actual_shape(*spec, actual_shape);
      const std::size_t expected_size = physicsnemo::inference::element_count(actual_shape) *
                                        physicsnemo::inference::dtype_size(spec->dtype);
      owned_inputs.emplace_back(spec->name, spec->dtype, physicsnemo::inference::Device{},
                                std::move(actual_shape),
                                read_bytes(file_input.path, expected_size));
      if (provided_shape != shapes_by_name.end()) {
        shapes_by_name.erase(provided_shape);
      }
    }
    if (!shapes_by_name.empty()) {
      throw std::invalid_argument("--input-shape has no matching --input-file: " +
                                  shapes_by_name.begin()->first);
    }
  }

  std::vector<physicsnemo::inference::TensorView> input_views;
  input_views.reserve(owned_inputs.size());
  for (const auto& input : owned_inputs) input_views.push_back(input.view());

  for (std::size_t i = 0; i < warmup; ++i) {
    static_cast<void>(session->run(input_views));
  }

  std::vector<double> durations_ms;
  durations_ms.reserve(iterations);
  std::vector<physicsnemo::inference::OwnedTensor> outputs;
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    outputs = session->run(input_views);
    durations_ms.push_back(
        elapsed_ms(start, std::chrono::steady_clock::now()));
  }

  if (benchmark_requested) {
    auto sorted = durations_ms;
    std::ranges::sort(sorted);
    const double mean =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    std::cout << std::fixed << std::setprecision(3)
              << "benchmark: warmup=" << warmup
              << " iterations=" << iterations
              << " package_load_ms=" << package_load_ms
              << " session_create_ms=" << session_create_ms
              << " mean_ms=" << mean << " p50_ms=" << percentile(sorted, 50)
              << " p95_ms=" << percentile(sorted, 95)
              << " min_ms=" << sorted.front()
              << " max_ms=" << sorted.back() << '\n';
    std::cout << std::defaultfloat << std::setprecision(6);
  }

  if (!output_files.empty()) {
    std::unordered_set<std::string> written_names;
    for (const auto& file_output : output_files) {
      const physicsnemo::inference::OwnedTensor* output = nullptr;
      if (file_output.name.empty()) {
        if (outputs.size() != 1 || output_files.size() != 1) {
          throw std::invalid_argument(
              "multi-output packages require named --output-file arguments");
        }
        output = &outputs.front();
      } else {
        const auto found = std::ranges::find_if(
            outputs, [&](const auto& candidate) {
              return candidate.view().name == file_output.name;
            });
        if (found == outputs.end()) {
          throw std::invalid_argument("unknown output tensor: " +
                                      file_output.name);
        }
        output = &*found;
      }
      const std::string name = output->view().name;
      if (!written_names.insert(name).second) {
        throw std::invalid_argument("duplicate output tensor: " + name);
      }
      write_bytes(file_output.path, output->storage());
      std::cout << name << ": wrote " << output->storage().size()
                << " bytes to " << file_output.path.string() << '\n';
    }
  } else {
    for (const auto& owned_output : outputs) {
      const auto output = owned_output.view();
      if (output.dtype != physicsnemo::inference::DType::kFloat32) {
        throw std::invalid_argument(
            "text output supports float32 tensors only; use --output-file");
      }
      std::vector<float> output_values(output.byte_size / sizeof(float));
      std::memcpy(output_values.data(), output.data, output.byte_size);
      std::cout << output.name << ':';
      for (const float value : output_values) std::cout << ' ' << value;
      std::cout << '\n';
    }
  }
  if (!output_metadata.empty()) {
    auto observed = nlohmann::json::array();
    for (const auto& owned_output : outputs) {
      const auto output = owned_output.view();
      observed.push_back({
          {"name", output.name},
          {"dtype", physicsnemo::inference::to_string(output.dtype)},
          {"shape", output.shape},
          {"device", {{"type", physicsnemo::inference::to_string(output.device.type)},
                      {"index", output.device.index}}},
          {"byte_size", output.byte_size},
      });
    }
    const nlohmann::json metadata = {
        {"schema_version", 1},
        {"backend", session->backend_name()},
        {"execution_device", {{"type", physicsnemo::inference::to_string(options.device.type)},
                              {"index", options.device.index}}},
        {"completed", true},
        {"outputs", std::move(observed)},
    };
    std::ofstream stream(output_metadata, std::ios::trunc);
    if (!stream) {
      throw std::runtime_error("cannot open output metadata: " + output_metadata.string());
    }
    stream << metadata.dump(2) << '\n';
    stream.close();
    if (!stream) {
      throw std::runtime_error("cannot write output metadata: " + output_metadata.string());
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "inspect" && argc == 3) return inspect(argv[2]);
    if (command == "run") return run(argc, argv);
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
