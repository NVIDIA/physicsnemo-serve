#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <limits>

namespace physicsnemo::inference::tensorrt_detail {

// Descriptor plumbing only: each plugin retains its shape contract and kernel.
inline bool element_count(const nvinfer1::Dims& input, std::int64_t* elements) {
  if (input.nbDims < 1) return false;
  std::int64_t result = 1;
  for (int32_t index = 0; index < input.nbDims; ++index) {
    if (input.d[index] <= 0 ||
        result > std::numeric_limits<std::int64_t>::max() / input.d[index]) {
      return false;
    }
    result *= input.d[index];
  }
  *elements = result;
  return true;
}

template <typename Plugin>
nvinfer1::IPluginCapability* capability_interface(
    Plugin* plugin, nvinfer1::PluginCapabilityType type) noexcept {
  switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
      return static_cast<nvinfer1::IPluginV3OneCore*>(plugin);
    case nvinfer1::PluginCapabilityType::kBUILD:
      return static_cast<nvinfer1::IPluginV3OneBuild*>(plugin);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(plugin);
  }
  return nullptr;
}

inline bool supports_fp32_linear(
    int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
    int32_t nb_inputs, int32_t nb_outputs, int32_t expected_inputs) noexcept {
  if (in_out == nullptr || position < 0 || position >= expected_inputs + 1 ||
      nb_inputs != expected_inputs || nb_outputs != 1) {
    return false;
  }
  const auto& descriptor = in_out[position].desc;
  return descriptor.type == nvinfer1::DataType::kFLOAT &&
         descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

}  // namespace physicsnemo::inference::tensorrt_detail
