#include <NvInfer.h>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "physicsnemo/inference/backends/tensorrt_exact_gemm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_deslice_bmm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_slice_bmm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_layer_norm_plugin.hpp"
#include "physicsnemo/inference/backends/tensorrt_exact_softmax_plugin.hpp"

namespace {

nvinfer1::PluginTensorDesc tensor(std::initializer_list<std::int64_t> shape) {
  nvinfer1::PluginTensorDesc result{};
  result.dims.nbDims = static_cast<int>(shape.size());
  int index = 0;
  for (const auto size : shape) result.dims.d[index++] = size;
  result.type = nvinfer1::DataType::kFLOAT;
  result.format = nvinfer1::TensorFormat::kLINEAR;
  return result;
}

int check_shape(const char* name,
                std::initializer_list<nvinfer1::PluginTensorDesc> inputs,
                bool accepted) {
  auto* creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(
      getPluginRegistry()->getCreator(name, "1", ""));
  if (creator == nullptr) throw std::runtime_error("Plugin creator is absent");
  nvinfer1::PluginFieldCollection fields{0, nullptr};
  std::unique_ptr<nvinfer1::IPluginV3> plugin(
      creator->createPlugin("dimensions", &fields, nvinfer1::TensorRTPhase::kRUNTIME));
  if (!plugin) throw std::runtime_error("Plugin creation failed");
  auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  if (runtime == nullptr) throw std::runtime_error("Runtime capability is absent");
  const bool actual = runtime->onShapeChange(
      inputs.begin(), static_cast<int>(inputs.size()), nullptr, 1) == 0;
  if (actual == accepted) return 0;
  std::cerr << name << " accepted=" << actual << ", expected=" << accepted << '\n';
  return 1;
}

}  // namespace

int main() {
  try {
    if (pnmir_tensorrt_exact_gemm_register() != 0 ||
        pnmir_tensorrt_exact_slice_bmm_register() != 0 ||
        pnmir_tensorrt_exact_deslice_bmm_register() != 0 ||
        pnmir_tensorrt_exact_layer_norm_register() != 0 ||
        pnmir_tensorrt_exact_softmax_register() != 0) {
      throw std::runtime_error("Plugin registration failed");
    }
    const auto too_large = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
    int failures = 0;
    failures += check_shape("PNMIRExactGemm", {tensor({1, 75, 256}),
                            tensor({256, 256})}, true);
    failures += check_shape("PNMIRExactGemm", {tensor({1, 75, 256}),
                            tensor({too_large, 256})}, false);
    failures += check_shape("PNMIRExactGemm", {tensor({1, 75, too_large}),
                            tensor({256, too_large})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, 8, 512}),
                            tensor({1, 75, 8, 32})}, true);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, too_large, 8, 512}),
                            tensor({1, too_large, 8, 32})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, too_large, 512}),
                            tensor({1, 75, too_large, 32})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, 8, too_large}),
                            tensor({1, 75, 8, 32})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, 8, 512}),
                            tensor({1, 75, 8, too_large})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, 65536, 65536}),
                            tensor({1, 75, 65536, 32})}, false);
    failures += check_shape("PNMIRExactSliceBmm", {tensor({1, 75, 65536, 512}),
                            tensor({1, 75, 65536, 65536})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 8, 512}),
                            tensor({1, 512, 8, 32})}, true);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 2048, 8, 512}),
                            tensor({1, 512, 8, 32})}, true);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 8, 512}),
                            tensor({1, 8, 512, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({2, 75, 8, 512}),
                            tensor({2, 512, 8, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 1, 512}),
                            tensor({1, 512, 1, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 0, 8, 512}),
                            tensor({1, 512, 8, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, too_large, 8, 512}),
                            tensor({1, 512, 8, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, too_large, 512}),
                            tensor({1, 512, too_large, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 8, too_large}),
                            tensor({1, too_large, 8, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 8, 512}),
                            tensor({1, 512, 8, too_large})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 65536, 65536}),
                            tensor({1, 65536, 65536, 32})}, false);
    failures += check_shape("PNMIRExactDesliceBmm", {tensor({1, 75, 65536, 512}),
                            tensor({1, 512, 65536, 65536})}, false);
    const std::int64_t wrapped_width = (std::int64_t{1} << 32) + 256;
    failures += check_shape("PNMIRExactLayerNorm", {tensor({1, 75, 256}),
                            tensor({256}), tensor({256})}, true);
    failures += check_shape("PNMIRExactLayerNorm", {tensor({1, 75, wrapped_width}),
                            tensor({256}), tensor({256})}, false);
    failures += check_shape("PNMIRExactLayerNorm", {tensor({1, 75, wrapped_width}),
                            tensor({wrapped_width}), tensor({wrapped_width})}, false);
    constexpr std::int64_t aligned_limit = 2147483644;
    failures += check_shape("PNMIRExactLayerNorm", {tensor({1, 1, aligned_limit}),
                            tensor({aligned_limit}), tensor({aligned_limit})}, true);
    failures += check_shape("PNMIRExactSoftmax", {tensor({1, 128})}, true);
    failures += check_shape("PNMIRExactSoftmax", {tensor({1, 512})}, true);
    failures += check_shape("PNMIRExactSoftmax",
                            {tensor({1, (std::int64_t{1} << 32) + 128})}, false);
    failures += check_shape("PNMIRExactSoftmax",
                            {tensor({1, (std::int64_t{1} << 32) + 512})}, false);
    if (failures != 0) return 1;
    std::cout << "Exact plugin dimension validation passed (30 cases)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
