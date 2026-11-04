#pragma once

#include <NvInfer.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <new>

namespace physicsnemo::inference::tensorrt_detail {

// Creator version 2 binds the qualified H100 provider's numerical contract.
// Version 1 plugins retain their original IEEE behavior and empty fields.
class ProviderMathPolicy {
 public:
  explicit ProviderMathPolicy(bool enabled = false) : enabled_(enabled) {
    for (std::size_t index = 0; index < fields_.size(); ++index) {
      fields_[index] = nvinfer1::PluginField{
          names_[index], &values_[index], nvinfer1::PluginFieldType::kINT32, 1};
    }
    collection_.nbFields = enabled_ ? static_cast<int32_t>(fields_.size()) : 0;
    collection_.fields = enabled_ ? fields_.data() : nullptr;
  }

  ProviderMathPolicy(const ProviderMathPolicy&) = delete;
  ProviderMathPolicy& operator=(const ProviderMathPolicy&) = delete;

  bool enabled() const noexcept { return enabled_; }
  const char* version() const noexcept { return enabled_ ? "2" : "1"; }
  std::size_t workspace_bytes() const noexcept {
    return enabled_ ? static_cast<std::size_t>(values_[1]) : 0;
  }
  std::size_t lt_workspace_bytes() const noexcept {
    return static_cast<std::size_t>(values_[2]);
  }
  const nvinfer1::PluginFieldCollection* fields() const noexcept {
    return &collection_;
  }

  static bool accepts(const nvinfer1::PluginFieldCollection* collection) noexcept {
    if (collection == nullptr || collection->nbFields != 3 ||
        collection->fields == nullptr) return false;
    std::array<bool, 3> seen{};
    for (int32_t index = 0; index < collection->nbFields; ++index) {
      const auto& field = collection->fields[index];
      if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
          field.type != nvinfer1::PluginFieldType::kINT32) return false;
      bool known = false;
      for (std::size_t target = 0; target < names_.size(); ++target) {
        if (std::strcmp(field.name, names_[target]) != 0) continue;
        int32_t value{};
        std::memcpy(&value, field.data, sizeof(value));
        if (seen[target] || value != expected_[target]) return false;
        seen[target] = true;
        known = true;
        break;
      }
      if (!known) return false;
    }
    return seen[0] && seen[1] && seen[2];
  }

 private:
  inline static constexpr std::array<const char*, 3> names_{
      "compute_mode", "cublas_workspace_bytes", "cublaslt_workspace_bytes"};
  // Pinned Torch 2.10: TF32, Hopper cuBLAS 32 MiB, cuBLASLt 1 MiB.
  inline static constexpr std::array<int32_t, 3> expected_{1, 33554432, 1048576};
  bool enabled_;
  std::array<int32_t, 3> values_{expected_};
  std::array<nvinfer1::PluginField, 3> fields_{};
  nvinfer1::PluginFieldCollection collection_{};
};

template <typename Plugin>
class ProviderPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
  explicit ProviderPluginCreator(const char* name) : name_(name) {}
  nvinfer1::IPluginV3* createPlugin(
      const char*, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase) noexcept override {
    if (!ProviderMathPolicy::accepts(fields)) return nullptr;
    return new (std::nothrow) Plugin(true);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return policy_.fields();
  }
  const char* getPluginName() const noexcept override { return name_; }
  const char* getPluginVersion() const noexcept override { return "2"; }
  const char* getPluginNamespace() const noexcept override { return ""; }

 private:
  const char* name_;
  ProviderMathPolicy policy_{true};
};

}  // namespace physicsnemo::inference::tensorrt_detail
