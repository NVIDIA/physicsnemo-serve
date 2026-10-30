#pragma once

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace physicsnemo::inference::tensorrt_detail {

inline bool configured_workspace_size(const char* config, std::size_t* bytes) {
  if (config == nullptr) return true;
  std::size_t total = 0;
  bool matched = false;
  // Match Torch's :SIZE_KIB:COUNT pairs, including multiple workspace pools.
  for (const char* cursor = config; *cursor != '\0';) {
    if (*cursor != ':' || cursor[1] < '0' || cursor[1] > '9') {
      ++cursor;
      continue;
    }
    char* size_end = nullptr;
    errno = 0;
    const auto size = std::strtoull(cursor + 1, &size_end, 10);
    const bool size_overflow = errno == ERANGE;
    if (*size_end != ':' || size_end[1] < '0' || size_end[1] > '9') {
      ++cursor;
      continue;
    }
    char* count_end = nullptr;
    errno = 0;
    const auto count = std::strtoull(size_end + 1, &count_end, 10);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (size_overflow || errno == ERANGE || size > limit / 1024U ||
        (size != 0 && count > (limit - total) / (size * 1024U))) {
      return false;
    }
    total += static_cast<std::size_t>(size * count * 1024U);
    matched = true;
    cursor = count_end;
  }
  if (matched) {
    *bytes = total;
  } else {
    std::fprintf(stderr,
                 "Ignoring invalid CUBLAS_WORKSPACE_CONFIG for exact BLAS\n");
  }
  return true;
}

inline bool gemm_workspace_size(std::size_t* bytes) {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return false;
  }
  // PyTorch 2.10 CublasHandlePool.cpp: Hopper uses 32 MiB, other CUDA
  // devices use :4096:2:16:8. This is distinct from the Lt 1 MiB workspace.
  *bytes = properties.major == 9 && properties.minor == 0
               ? 32U * 1024U * 1024U
               : 4096U * 1024U * 2U + 16U * 1024U * 8U;
#if defined(_MSC_VER)
  char* config = nullptr;
  std::size_t length = 0;
  if (_dupenv_s(&config, &length, "CUBLAS_WORKSPACE_CONFIG") != 0) return false;
  const bool valid = configured_workspace_size(config, bytes);
  std::free(config);
  return valid;
#else
  return configured_workspace_size(std::getenv("CUBLAS_WORKSPACE_CONFIG"), bytes);
#endif
}

}  // namespace physicsnemo::inference::tensorrt_detail
