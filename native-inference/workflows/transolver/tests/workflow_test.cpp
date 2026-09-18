#include "transolver_workflow/workflow.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void check_throws(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

std::filesystem::path write_stats_fixture() {
  const auto suffix = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("transolver-workflow-stats-" +
                     std::to_string(suffix) + ".json");
  std::ofstream stream(path);
  stream << R"({
    "mean": {
      "pressure": [10.0],
      "shear_stress": [1.0, 2.0, 3.0],
      "velocity": [4.0, 5.0, 6.0],
      "turbulent_viscosity": [7.0]
    },
    "std_dev": {
      "pressure": [2.0],
      "shear_stress": [3.0, 4.0, 5.0],
      "velocity": [6.0, 7.0, 8.0],
      "turbulent_viscosity": [9.0]
    }
  })";
  stream.close();
  return path;
}

std::string base64_encode(const std::vector<std::uint8_t>& bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded;
  encoded.reserve(((bytes.size() + 2U) / 3U) * 4U);
  for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
    const std::size_t remaining = bytes.size() - offset;
    const std::uint32_t value =
        static_cast<std::uint32_t>(bytes[offset]) << 16U |
        static_cast<std::uint32_t>(remaining > 1U ? bytes[offset + 1U] : 0U)
            << 8U |
        static_cast<std::uint32_t>(remaining > 2U ? bytes[offset + 2U] : 0U);
    encoded.push_back(alphabet[(value >> 18U) & 63U]);
    encoded.push_back(alphabet[(value >> 12U) & 63U]);
    encoded.push_back(remaining > 1U ? alphabet[(value >> 6U) & 63U] : '=');
    encoded.push_back(remaining > 2U ? alphabet[value & 63U] : '=');
  }
  return encoded;
}

std::filesystem::path write_inline_binary_vtu_fixture(
    const bool use_uint64_header) {
  const auto suffix = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("transolver-workflow-points-" +
                     std::to_string(suffix) +
                     (use_uint64_header ? "-u64.vtu" : "-u32.vtu"));
  const std::array<float, 12> points{
      1.0F, 2.0F, 3.0F,
      -4.0F, 5.5F, 6.25F,
      7.0F, -8.0F, 9.0F,
      10.0F, 11.0F, -12.0F,
  };
  const std::uint64_t payload_bytes = points.size() * sizeof(float);
  const std::size_t header_bytes =
      use_uint64_header ? sizeof(std::uint64_t) : sizeof(std::uint32_t);
  std::vector<std::uint8_t> binary(header_bytes + payload_bytes);
  std::memcpy(binary.data(), &payload_bytes, header_bytes);
  std::memcpy(
      binary.data() + header_bytes,
      points.data(),
      static_cast<std::size_t>(payload_bytes));

  std::ofstream stream(path, std::ios::binary);
  stream
      << "<?xml version='1.0'?>\n"
      << "<VTKFile type='UnstructuredGrid' version='0.1' "
         "byte_order='LittleEndian' header_type='"
      << (use_uint64_header ? "UInt64" : "UInt32") << "'>\n"
      << "<UnstructuredGrid><Piece NumberOfPoints='4' NumberOfCells='0'>\n"
      << "<Points><DataArray type='Float32' Name='Points' "
         "NumberOfComponents='3' format='binary'>\n"
      << base64_encode(binary)
      << "\n</DataArray></Points></Piece></UnstructuredGrid></VTKFile>\n";
  stream.close();
  return path;
}

void test_domains_and_contracts() {
  using transolver_workflow::Domain;
  check(
      transolver_workflow::parse_domain("surface") == Domain::kSurface,
      "surface domain parsing failed");
  check(
      transolver_workflow::parse_domain("volume") == Domain::kVolume,
      "volume domain parsing failed");
  check_throws(
      [] { transolver_workflow::parse_domain("other"); },
      "unknown domain was accepted");

  physicsnemo::inference::ModelManifest surface;
  surface.inputs = {
      {"fx", physicsnemo::inference::DType::kFloat32, {1, -1, 2}},
      {"embedding", physicsnemo::inference::DType::kFloat32, {1, -1, 6}},
  };
  surface.outputs = {
      {"surface_fields_standardized", physicsnemo::inference::DType::kFloat32, {1, -1, 4}},
  };
  transolver_workflow::validate_package_contract(
      surface, Domain::kSurface, 2048);

  auto fixed_output = surface;
  fixed_output.outputs[0].shape[1] = 2048;
  check_throws(
      [&] {
        transolver_workflow::validate_package_contract(
            fixed_output, Domain::kSurface, 32);
      },
      "fixed-size output was accepted for a different block size");

  auto volume = surface;
  volume.inputs[1].shape[2] = 7;
  volume.outputs[0].shape[2] = 5;
  transolver_workflow::validate_package_contract(
      volume, Domain::kVolume, 2048);
  check_throws(
      [&] {
        transolver_workflow::validate_package_contract(
            surface, Domain::kVolume, 2048);
      },
      "surface package was accepted for volume");
  check_throws(
      [&] {
        transolver_workflow::validate_package_contract(
            surface, Domain::kSurface, 1);
      },
      "one-point block was accepted");
}

void test_stats_and_decoding() {
  using transolver_workflow::Domain;
  const auto path = write_stats_fixture();
  const torch::Device cpu(torch::kCPU);
  const auto surface_stats =
      transolver_workflow::load_normalization_stats(
          path, Domain::kSurface, cpu);
  check(
      torch::equal(
          surface_stats.mean,
          torch::tensor({10.0F, 1.0F, 2.0F, 3.0F})),
      "surface means use the wrong channel order");
  check(
      torch::equal(
          surface_stats.standard_deviation,
          torch::tensor({2.0F, 3.0F, 4.0F, 5.0F})),
      "surface deviations use the wrong channel order");
  const auto surface = transolver_workflow::decode_physical_outputs(
      torch::zeros({1, 1, 4}),
      surface_stats,
      Domain::kSurface,
      2.0,
      3.0,
      0.0);
  check(
      torch::equal(
          surface,
          torch::tensor({180.0F, 18.0F, 36.0F, 54.0F})
              .reshape({1, 1, 4})),
      "surface physical decoding is incorrect");

  const auto volume_stats =
      transolver_workflow::load_normalization_stats(
          path, Domain::kVolume, cpu);
  check(
      torch::equal(
          volume_stats.mean,
          torch::tensor({4.0F, 5.0F, 6.0F, 10.0F, 7.0F})),
      "volume means use the wrong channel order");
  const auto volume = transolver_workflow::decode_physical_outputs(
      torch::zeros({1, 1, 5}),
      volume_stats,
      Domain::kVolume,
      2.0,
      3.0,
      4.0);
  check(
      torch::equal(
          volume,
          torch::tensor({12.0F, 15.0F, 18.0F, 180.0F, 84.0F})
              .reshape({1, 1, 5})),
      "volume physical decoding is incorrect");
  std::filesystem::remove(path);
}

void test_bounded_inline_binary_vtu_reader() {
  const auto path = write_inline_binary_vtu_fixture(true);
  const torch::Device cpu(torch::kCPU);
  const auto prefix = transolver_workflow::read_inline_binary_vtu_points(
      path, 2, cpu);
  check(
      torch::equal(
          prefix,
          torch::tensor(
              {{1.0F, 2.0F, 3.0F}, {-4.0F, 5.5F, 6.25F}})),
      "bounded VTU reader returned the wrong point prefix");
  const auto complete = transolver_workflow::read_inline_binary_vtu_points(
      path, 0, cpu);
  check(complete.sizes().vec() == std::vector<std::int64_t>({4, 3}),
        "unbounded VTU reader returned the wrong shape");
  check(
      torch::equal(
          complete.index({3}), torch::tensor({10.0F, 11.0F, -12.0F})),
      "unbounded VTU reader returned the wrong final point");
  transolver_workflow::InlineBinaryVtuPointStream blocks(path);
  check(blocks.total_points() == 4 && blocks.point_offset() == 0,
        "VTU stream metadata mismatch");
  const auto first = blocks.read_next(3, cpu);
  const auto last = blocks.read_next(1, cpu);
  check(torch::equal(first, complete.slice(0, 0, 3)) &&
            torch::equal(last, complete.slice(0, 3, 4)) &&
            blocks.point_offset() == 4,
        "VTU sequential block reader mismatch");
  check_throws(
      [&] { static_cast<void>(blocks.read_next(1, cpu)); },
      "VTU stream accepted a read past the payload");
  check_throws(
      [&] {
        transolver_workflow::read_inline_binary_vtu_points(path, 5, cpu);
      },
      "VTU reader accepted a point limit beyond NumberOfPoints");
  const auto uint32_path = write_inline_binary_vtu_fixture(false);
  const auto uint32_points =
      transolver_workflow::read_inline_binary_vtu_points(
          uint32_path, 0, cpu);
  check(
      torch::equal(uint32_points, complete),
      "UInt32 VTU binary header decoding differs from UInt64");
  std::filesystem::remove(path);
  std::filesystem::remove(uint32_path);
}

}  // namespace

int main() {
  try {
    test_domains_and_contracts();
    test_stats_and_decoding();
    test_bounded_inline_binary_vtu_reader();
    std::cout << "transolver workflow tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "transolver workflow test failure: " << error.what() << '\n';
    return 1;
  }
}
