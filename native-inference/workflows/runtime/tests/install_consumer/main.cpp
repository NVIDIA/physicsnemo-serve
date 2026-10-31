#include "physicsnemo/workflow/v1/host.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
  try {
    if (argc != 2) return 2;
    std::ifstream stream(argv[1]);
    const std::string config{std::istreambuf_iterator<char>(stream),{}};
    physicsnemo::workflow::v1::WorkflowHost host(config);
    const auto result = host.self_test();
    if (result.find("\"status\":\"passed\"") == std::string::npos) return 3;
    std::cout << result << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
