#include "transolver_workflow/warp_sdf.hpp"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#define WP_ENABLE_CUDA 1
#include <warp.h>

namespace transolver_workflow {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpLbvhConstructor = 2;
constexpr int kWarpBvhLeafSize = 4;

void check_cuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

void initialize_warp() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    if (wp_init(PNMIR_WARP_VERSION) != 0) {
      throw std::runtime_error(
          std::string("Warp initialization failed: ") + wp_get_error_string());
    }
    if (wp_is_cuda_enabled() == 0) {
      throw std::runtime_error("Warp runtime was built without CUDA support");
    }
  });
}

__global__ void signed_distance_kernel(
    const wp::uint64 mesh_id,
    const wp::vec3* points,
    const std::int64_t point_count,
    wp::float32* distance,
    wp::vec3* closest_point) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= point_count) {
    return;
  }
  const auto result = wp::mesh_query_point_sign_winding_number(
      mesh_id, points[index], 1.0e8F, 2.0F, 0.5F);
  const auto mesh = wp::mesh_get(mesh_id);
  const auto p0 = mesh.points[mesh.indices[3 * result.face]];
  const auto p1 = mesh.points[mesh.indices[3 * result.face + 1]];
  const auto p2 = mesh.points[mesh.indices[3 * result.face + 2]];
  const auto hit = result.u * p0 + result.v * p1 +
                   (1.0F - result.u - result.v) * p2;
  distance[index] =
      result.sign * wp::abs(wp::length(points[index] - hit));
  closest_point[index] = hit;
}

void validate_tensor(
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const std::int64_t rank,
    const char* name) {
  if (!tensor.is_cuda() || tensor.scalar_type() != dtype ||
      tensor.dim() != rank || !tensor.is_contiguous()) {
    throw std::invalid_argument(
        std::string(name) +
        " must be a contiguous CUDA tensor with the required dtype and rank");
  }
}

class WarpMesh final {
 public:
  WarpMesh(
      void* context,
      const torch::Tensor& vertices,
      const torch::Tensor& indices)
      : id_(wp_mesh_create_device(
            context,
            wp::array_t<wp::vec3>(
                reinterpret_cast<wp::vec3*>(vertices.data_ptr<float>()),
                static_cast<int>(vertices.size(0))),
            wp::array_t<wp::vec3>(),
            wp::array_t<wp::int32>(
                reinterpret_cast<wp::int32*>(
                    indices.data_ptr<std::int32_t>()),
                static_cast<int>(indices.numel())),
            static_cast<int>(vertices.size(0)),
            static_cast<int>(indices.numel() / 3),
            1,
            kWarpLbvhConstructor,
            nullptr,
            kWarpBvhLeafSize)) {
    if (id_ == 0) {
      throw std::runtime_error(
          std::string("Warp mesh creation failed: ") + wp_get_error_string());
    }
  }

  ~WarpMesh() {
    if (id_ != 0) {
      wp_mesh_destroy_device(id_);
    }
  }

  WarpMesh(const WarpMesh&) = delete;
  WarpMesh& operator=(const WarpMesh&) = delete;

  std::uint64_t id() const { return id_; }

 private:
  std::uint64_t id_;
};

}  // namespace

SignedDistanceResult warp_signed_distance(
    const torch::Tensor& mesh_vertices,
    const torch::Tensor& mesh_indices,
    const torch::Tensor& query_points) {
  initialize_warp();
  validate_tensor(
      mesh_vertices, torch::kFloat32, 2, "mesh_vertices");
  validate_tensor(mesh_indices, torch::kInt32, 1, "mesh_indices");
  validate_tensor(query_points, torch::kFloat32, 2, "query_points");
  if (mesh_vertices.size(1) != 3 || query_points.size(1) != 3 ||
      mesh_indices.numel() == 0 || mesh_indices.numel() % 3 != 0) {
    throw std::invalid_argument("Warp SDF tensor shapes are incompatible");
  }
  if (mesh_vertices.get_device() != query_points.get_device() ||
      mesh_indices.get_device() != query_points.get_device()) {
    throw std::invalid_argument("Warp SDF tensors must share one CUDA device");
  }
  if (mesh_vertices.size(0) > std::numeric_limits<int>::max() ||
      mesh_indices.numel() > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Warp SDF mesh exceeds the int32 index contract");
  }

  const int device_index = query_points.get_device();
  void* context = wp_cuda_device_get_primary_context(device_index);
  if (context == nullptr) {
    throw std::runtime_error("Warp could not obtain the CUDA primary context");
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  wp_cuda_context_set_stream(context, stream.stream(), 0);

  auto distance = torch::empty(
      {query_points.size(0)}, query_points.options());
  auto closest_point = torch::empty_like(query_points);
  WarpMesh mesh(context, mesh_vertices, mesh_indices);
  const auto blocks = static_cast<unsigned int>(
      (query_points.size(0) + kThreads - 1) / kThreads);
  signed_distance_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
      mesh.id(),
      reinterpret_cast<const wp::vec3*>(query_points.const_data_ptr<float>()),
      query_points.size(0),
      distance.data_ptr<float>(),
      reinterpret_cast<wp::vec3*>(closest_point.data_ptr<float>()));
  check_cuda(cudaGetLastError(), "Warp-compatible SDF kernel launch");
  check_cuda(
      cudaStreamSynchronize(stream.stream()),
      "Warp-compatible SDF synchronization");
  return {std::move(distance), std::move(closest_point)};
}

}  // namespace transolver_workflow
