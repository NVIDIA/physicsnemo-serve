#include "physicsnemo/workflow/v1/host.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

std::string read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read JSON file: " + path.string());
  std::string value;
  char chunk[8192];
  while (stream.read(chunk, sizeof(chunk)) || stream.gcount()) {
    value.append(chunk, static_cast<std::size_t>(stream.gcount()));
    if (value.size() > 64 * 1024 * 1024)
      throw std::runtime_error("JSON file exceeds 64 MiB: " + path.string());
  }
  if (!stream.eof()) throw std::runtime_error("cannot read complete JSON file: " + path.string());
  return value;
}

void write(const std::filesystem::path& path, const std::string& value) {
  if (std::filesystem::exists(path))
    throw std::runtime_error("output file already exists: " + path.string());
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot write JSON file: " + path.string());
  stream << value << '\n';
  stream.close();
  if (!stream) throw std::runtime_error("cannot write complete JSON file: " + path.string());
}

constexpr const char* usage =
    "usage: pnms-native-run --config FILE (--request FILE | --self-test) "
    "[--output FILE] [--startup-report FILE]\n";
}

int main(int argc, char** argv) {
  std::string config, request, output, startup_report;
  bool self_test = false;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--help" || option == "-h") {
      std::cout << usage;
      return 0;
    }
    if (option == "--self-test" && !self_test) {
      self_test = true;
      continue;
    }
    std::string* destination = nullptr;
    if (option == "--config") destination = &config;
    else if (option == "--request") destination = &request;
    else if (option == "--output") destination = &output;
    else if (option == "--startup-report") destination = &startup_report;
    if (!destination || !destination->empty() || index + 1 >= argc) {
      std::cerr << usage;
      return 2;
    }
    *destination = argv[++index];
  }
  if (config.empty() || (request.empty() == !self_test)) {
    std::cerr << usage;
    return 2;
  }
  try {
    std::signal(SIGINT, interrupt);
    std::signal(SIGTERM, interrupt);
    const auto cancellation = [] { return interrupted != 0; };
    // Construction, startup checks, execution and destruction all stay on this
    // OS thread, exactly as they do behind the Rust worker's native bridge.
    physicsnemo::workflow::v1::WorkflowHost host(read(config));
    const auto startup = host.self_test(cancellation);
    if (!startup_report.empty()) write(startup_report, startup);
    const auto result = self_test ? startup : host.run(read(request), cancellation);
    if (!output.empty()) write(output, result);
    else std::cout << result << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pnms-native-run: " << error.what() << '\n';
    return 1;
  }
}
