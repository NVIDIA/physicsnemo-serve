#include "transolver_workflow/workflow.hpp"

#include <ATen/ops/linalg_vector_norm.h>

#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <vtkCellArray.h>
#include <vtkCellCenters.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkTriangleFilter.h>
#include <vtkXMLPolyDataReader.h>

#include "transolver_workflow/warp_sdf.hpp"

namespace transolver_workflow {
namespace {

using Json = nlohmann::json;

struct StlGeometry {
  torch::Tensor coordinates;
  torch::Tensor triangle_indices;
  torch::Tensor centers;
};

torch::Tensor vector_norm(const torch::Tensor& input) {
  return at::linalg_vector_norm(
      input, 2.0, std::vector<std::int64_t>{-1}, true, std::nullopt);
}

std::int64_t resolved_count(
    const std::int64_t available,
    const std::int64_t point_limit,
    const std::string& label) {
  if (available <= 0) {
    throw std::runtime_error(label + " contains no inference locations");
  }
  if (point_limit < 0) {
    throw std::invalid_argument("point_limit cannot be negative");
  }
  if (point_limit == 0) {
    return static_cast<std::int64_t>(available);
  }
  if (point_limit > available) {
    throw std::invalid_argument(
        "point_limit exceeds the number of " + label + " locations");
  }
  return point_limit;
}

torch::Tensor points_to_tensor(
    vtkPoints* points,
    const std::int64_t count,
    const torch::Device& device) {
  if (points == nullptr || count > points->GetNumberOfPoints()) {
    throw std::runtime_error("VTK point array is missing or truncated");
  }
  std::vector<float> values(static_cast<std::size_t>(count) * 3U);
  for (std::int64_t index = 0; index < count; ++index) {
    double point[3]{};
    points->GetPoint(static_cast<vtkIdType>(index), point);
    const auto offset = static_cast<std::size_t>(index) * 3U;
    values[offset] = static_cast<float>(point[0]);
    values[offset + 1U] = static_cast<float>(point[1]);
    values[offset + 2U] = static_cast<float>(point[2]);
  }
  return torch::from_blob(
             values.data(),
             {count, 3},
             torch::TensorOptions().dtype(torch::kFloat32))
      .clone()
      .to(device);
}

torch::Tensor normalized_cell_normals(
    vtkDataArray* normals,
    const std::int64_t count,
    const torch::Device& device) {
  if (normals == nullptr || normals->GetNumberOfComponents() != 3 ||
      count > normals->GetNumberOfTuples()) {
    throw std::runtime_error("VTK surface cell normals are missing or truncated");
  }
  std::vector<float> values(static_cast<std::size_t>(count) * 3U);
  for (std::int64_t index = 0; index < count; ++index) {
    double tuple[3]{};
    normals->GetTuple(static_cast<vtkIdType>(index), tuple);
    float x = static_cast<float>(tuple[0]);
    float y = static_cast<float>(tuple[1]);
    float z = static_cast<float>(tuple[2]);
    const float length = std::sqrt(x * x + y * y + z * z) + 1.0e-8F;
    x /= length;
    y /= length;
    z /= length;
    const auto offset = static_cast<std::size_t>(index) * 3U;
    values[offset] = x;
    values[offset + 1U] = y;
    values[offset + 2U] = z;
  }
  return torch::from_blob(
             values.data(),
             {count, 3},
             torch::TensorOptions().dtype(torch::kFloat32))
      .clone()
      .to(device);
}

vtkSmartPointer<vtkPolyData> read_stl_polydata(
    const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("STL file does not exist: " + path.string());
  }
  vtkNew<vtkSTLReader> reader;
  reader->SetFileName(path.string().c_str());
  reader->MergingOn();
  reader->Update();

  vtkNew<vtkTriangleFilter> triangles;
  triangles->SetInputConnection(reader->GetOutputPort());
  triangles->Update();
  auto output = vtkSmartPointer<vtkPolyData>::New();
  output->ShallowCopy(triangles->GetOutput());
  if (output->GetNumberOfPoints() == 0 || output->GetNumberOfPolys() == 0) {
    throw std::runtime_error("STL geometry is empty: " + path.string());
  }
  return output;
}

StlGeometry load_stl_geometry(
    const std::filesystem::path& path,
    const torch::Device& device) {
  const auto mesh = read_stl_polydata(path);
  const auto point_count = static_cast<std::int64_t>(mesh->GetNumberOfPoints());
  const auto triangle_count =
      static_cast<std::int64_t>(mesh->GetNumberOfPolys());
  auto coordinates = points_to_tensor(mesh->GetPoints(), point_count, device);

  std::vector<std::int32_t> indices;
  indices.reserve(static_cast<std::size_t>(triangle_count) * 3U);
  vtkNew<vtkIdList> ids;
  auto* polys = mesh->GetPolys();
  polys->InitTraversal();
  while (polys->GetNextCell(ids)) {
    if (ids->GetNumberOfIds() != 3) {
      throw std::runtime_error("triangulated STL contains a non-triangle cell");
    }
    for (vtkIdType local = 0; local < 3; ++local) {
      const auto value = ids->GetId(local);
      if (value < 0 ||
          value > static_cast<vtkIdType>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("STL point index does not fit int32");
      }
      indices.push_back(static_cast<std::int32_t>(value));
    }
  }
  if (indices.size() != static_cast<std::size_t>(triangle_count) * 3U) {
    throw std::runtime_error("STL polygon traversal produced the wrong face count");
  }
  auto triangle_indices =
      torch::from_blob(
          indices.data(),
          {static_cast<std::int64_t>(indices.size())},
          torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(device);

  vtkNew<vtkCellCenters> center_filter;
  center_filter->SetInputData(mesh);
  center_filter->VertexCellsOff();
  center_filter->Update();
  auto centers = points_to_tensor(
      center_filter->GetOutput()->GetPoints(), triangle_count, device);
  return {
      std::move(coordinates),
      std::move(triangle_indices),
      std::move(centers),
  };
}

torch::Tensor load_surface_features(
    const std::filesystem::path& path,
    const std::int64_t point_limit,
    const torch::Device& device) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("VTP file does not exist: " + path.string());
  }
  vtkNew<vtkXMLPolyDataReader> reader;
  reader->SetFileName(path.string().c_str());
  reader->Update();
  vtkPolyData* mesh = reader->GetOutput();
  if (mesh == nullptr) {
    throw std::runtime_error("VTK did not produce a PolyData surface");
  }

  vtkSmartPointer<vtkPolyData> normalized_mesh;
  if (mesh->GetCellData()->GetNormals() == nullptr) {
    vtkNew<vtkPolyDataNormals> normal_filter;
    normal_filter->SetInputData(mesh);
    normal_filter->ComputeCellNormalsOn();
    normal_filter->ComputePointNormalsOff();
    normal_filter->SplittingOff();
    normal_filter->ConsistencyOn();
    normal_filter->AutoOrientNormalsOff();
    normal_filter->FlipNormalsOff();
    normal_filter->NonManifoldTraversalOn();
    normal_filter->Update();
    normalized_mesh = vtkSmartPointer<vtkPolyData>::New();
    normalized_mesh->ShallowCopy(normal_filter->GetOutput());
    mesh = normalized_mesh;
  }

  const auto count =
      resolved_count(mesh->GetNumberOfCells(), point_limit, "surface");
  vtkNew<vtkCellCenters> center_filter;
  center_filter->SetInputData(mesh);
  center_filter->VertexCellsOff();
  center_filter->Update();
  auto centers =
      points_to_tensor(center_filter->GetOutput()->GetPoints(), count, device);
  auto normals = normalized_cell_normals(
      mesh->GetCellData()->GetNormals(), count, device);
  normals = normals / vector_norm(normals);
  return torch::cat({centers, normals}, -1);
}

constexpr std::size_t kMaximumXmlTagBytes = 1024U * 1024U;

bool skip_to_xml_delimiter(
    std::ifstream& stream, const std::string_view delimiter) {
  std::array<char, 64U * 1024U> buffer;
  while (stream) {
    const auto position = stream.tellg();
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto size = static_cast<std::size_t>(stream.gcount());
    if (stream.bad()) {
      throw std::runtime_error("cannot read VTU XML");
    }
    const auto found = std::string_view(buffer.data(), size).find(delimiter);
    if (found != std::string_view::npos) {
      stream.clear();
      stream.seekg(
          position + static_cast<std::streamoff>(found + delimiter.size()));
      return static_cast<bool>(stream);
    }
    if (stream.eof()) {
      return false;
    }
    // Preserve delimiters split across two blocks without retaining payloads.
    if (delimiter.size() > 1U) {
      stream.seekg(-static_cast<std::streamoff>(delimiter.size() - 1U),
                   std::ios::cur);
    }
  }
  return false;
}

std::string next_xml_tag(std::ifstream& stream) {
  while (skip_to_xml_delimiter(stream, "<")) {
    std::string tag(1, '<');
    char value = '\0';
    while (stream.get(value)) {
      tag.push_back(value);
      if (tag == "<!--" || tag == "<![CDATA[") {
        if (!skip_to_xml_delimiter(stream, tag == "<!--" ? "-->" : "]]>")) {
          throw std::runtime_error("VTU contains truncated XML content");
        }
        break;
      }
      if (value == '>') {
        return tag;
      }
      if (tag.size() > kMaximumXmlTagBytes) {
        throw std::runtime_error("VTU XML tag exceeds the supported size");
      }
    }
    if (!stream) {
      throw std::runtime_error("VTU contains a truncated XML tag");
    }
  }
  throw std::runtime_error("VTU ended before its expected XML tag");
}

std::optional<std::string> xml_attribute(
    const std::string_view tag,
    const std::string_view name) {
  std::size_t cursor = 0;
  while ((cursor = tag.find(name, cursor)) != std::string_view::npos) {
    const bool starts_attribute =
        cursor == 0 ||
        std::isspace(static_cast<unsigned char>(tag[cursor - 1])) != 0 ||
        tag[cursor - 1] == '<';
    std::size_t equals = cursor + name.size();
    while (equals < tag.size() &&
           std::isspace(static_cast<unsigned char>(tag[equals])) != 0) {
      ++equals;
    }
    if (starts_attribute && equals < tag.size() && tag[equals] == '=') {
      ++equals;
      while (equals < tag.size() &&
             std::isspace(static_cast<unsigned char>(tag[equals])) != 0) {
        ++equals;
      }
      if (equals >= tag.size() ||
          (tag[equals] != '\'' && tag[equals] != '"')) {
        throw std::runtime_error(
            "VTU attribute is not quoted: " + std::string(name));
      }
      const char quote = tag[equals++];
      const auto end = tag.find(quote, equals);
      if (end == std::string_view::npos) {
        throw std::runtime_error(
            "VTU attribute has no closing quote: " + std::string(name));
      }
      return std::string(tag.substr(equals, end - equals));
    }
    cursor += name.size();
  }
  return std::nullopt;
}

std::uint64_t parse_unsigned_attribute(
    const std::string_view tag,
    const std::string_view name) {
  const auto text = xml_attribute(tag, name);
  if (!text.has_value() || text->empty()) {
    throw std::runtime_error("VTU is missing attribute: " + std::string(name));
  }
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(
      text->data(), text->data() + text->size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text->data() + text->size()) {
    throw std::runtime_error(
        "VTU attribute is not an unsigned integer: " + std::string(name));
  }
  return result;
}

class Base64StreamDecoder {
 public:
  explicit Base64StreamDecoder(std::ifstream& stream) : stream_(stream) {}

  void read_exact(std::uint8_t* destination, std::size_t bytes) {
    while (bytes != 0) {
      if (decoded_offset_ == decoded_size_) {
        decode_group();
      }
      const auto available = decoded_size_ - decoded_offset_;
      const auto copied = std::min(bytes, available);
      std::memcpy(destination, decoded_.data() + decoded_offset_, copied);
      destination += copied;
      bytes -= copied;
      decoded_offset_ += copied;
    }
  }

 private:
  static int decode_symbol(const char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    if (value == '=') return -2;
    return -1;
  }

  char next_encoded_character() {
    while (true) {
      if (encoded_offset_ == encoded_size_) {
        stream_.read(
            encoded_.data(),
            static_cast<std::streamsize>(encoded_.size()));
        encoded_size_ = static_cast<std::size_t>(stream_.gcount());
        encoded_offset_ = 0;
        if (encoded_size_ == 0) {
          throw std::runtime_error("VTU binary Points payload is truncated");
        }
      }
      const char value = encoded_[encoded_offset_++];
      if (std::isspace(static_cast<unsigned char>(value)) == 0) {
        return value;
      }
    }
  }

  void decode_group() {
    std::array<int, 4> symbols{};
    for (auto& symbol : symbols) {
      const char value = next_encoded_character();
      symbol = decode_symbol(value);
      if (symbol == -1) {
        throw std::runtime_error("VTU Points payload contains invalid base64");
      }
    }
    if (symbols[0] < 0 || symbols[1] < 0 ||
        (symbols[2] == -2 && symbols[3] != -2)) {
      throw std::runtime_error(
          "VTU Points payload contains invalid base64 padding");
    }
    const std::uint32_t value =
        (static_cast<std::uint32_t>(symbols[0]) << 18U) |
        (static_cast<std::uint32_t>(symbols[1]) << 12U) |
        (static_cast<std::uint32_t>(std::max(symbols[2], 0)) << 6U) |
        static_cast<std::uint32_t>(std::max(symbols[3], 0));
    decoded_[0] = static_cast<std::uint8_t>(value >> 16U);
    decoded_[1] = static_cast<std::uint8_t>(value >> 8U);
    decoded_[2] = static_cast<std::uint8_t>(value);
    decoded_size_ = symbols[2] == -2 ? 1U : symbols[3] == -2 ? 2U : 3U;
    decoded_offset_ = 0;
  }

  std::ifstream& stream_;
  std::array<char, 256U * 1024U> encoded_{};
  std::size_t encoded_offset_{0};
  std::size_t encoded_size_{0};
  std::array<std::uint8_t, 3> decoded_{};
  std::size_t decoded_offset_{0};
  std::size_t decoded_size_{0};
};

std::uint64_t little_endian_unsigned(
    const std::uint8_t* bytes,
    const std::size_t width) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < width; ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  return value;
}

std::vector<float> json_vector(
    const Json& block,
    const std::string& key,
    const std::size_t required) {
  if (!block.contains(key) || !block.at(key).is_array() ||
      block.at(key).size() < required) {
    throw std::invalid_argument(
        "normalization statistics are missing " + key);
  }
  std::vector<float> result;
  result.reserve(required);
  for (std::size_t index = 0; index < required; ++index) {
    result.push_back(static_cast<float>(block.at(key).at(index).get<double>()));
  }
  return result;
}

void append_prefix(
    std::vector<float>& destination,
    const Json& block,
    const std::string& key,
    const std::size_t required) {
  auto values = json_vector(block, key, required);
  destination.insert(destination.end(), values.begin(), values.end());
}

const physicsnemo::inference::TensorSpec& tensor_spec(
    const std::vector<physicsnemo::inference::TensorSpec>& specs,
    const std::string& name) {
  for (const auto& spec : specs) {
    if (spec.name == name) {
      return spec;
    }
  }
  throw std::invalid_argument("package is missing tensor: " + name);
}

void validate_tensor(
    const physicsnemo::inference::TensorSpec& spec,
    const std::int64_t width,
    const std::int64_t block_size,
    const std::string& label) {
  if (spec.dtype != physicsnemo::inference::DType::kFloat32 || spec.shape.size() != 3U ||
      spec.shape[0] != 1 || spec.shape[2] != width) {
    throw std::invalid_argument(label + " has an incompatible tensor contract");
  }
  if (spec.shape[1] != -1 && spec.shape[1] != block_size) {
    throw std::invalid_argument(
        label + " point dimension does not accept the configured block size");
  }
}

}  // namespace

class InlineBinaryVtuPointStream::Impl {
 public:
  explicit Impl(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument("VTU file does not exist: " + path.string());
    }
    if (!stream_) {
      throw std::runtime_error("cannot open VTU file: " + path.string());
    }
    if constexpr (std::endian::native != std::endian::little) {
      throw std::runtime_error(
          "inline binary VTU loading requires a little-endian host");
    }

    std::optional<std::string> vtk_file_tag;
    std::optional<std::string> piece_tag;
    std::optional<std::string> points_data_tag;
    bool inside_points = false;
    while (!points_data_tag.has_value()) {
      const auto tag = next_xml_tag(stream_);
      if (tag.starts_with("<VTKFile")) {
        vtk_file_tag = tag;
      } else if (tag.starts_with("<Piece")) {
        if (piece_tag.has_value()) {
          throw std::runtime_error(
              "bounded VTU loading supports exactly one Piece");
        }
        piece_tag = tag;
      } else if (tag.starts_with("<Points")) {
        inside_points = true;
      } else if (tag.starts_with("</Points")) {
        throw std::runtime_error("VTU Points element contains no DataArray");
      } else if (inside_points && tag.starts_with("<DataArray")) {
        points_data_tag = tag;
      }
    }
    if (!vtk_file_tag.has_value() || !piece_tag.has_value()) {
      throw std::runtime_error("VTU is missing VTKFile or Piece metadata");
    }
    if (xml_attribute(*vtk_file_tag, "type") != "UnstructuredGrid") {
      throw std::runtime_error("VTU VTKFile type must be UnstructuredGrid");
    }
    if (const auto compressor = xml_attribute(*vtk_file_tag, "compressor");
        compressor.has_value() && !compressor->empty()) {
      throw std::runtime_error(
          "bounded VTU loading does not support compressed Points data");
    }
    if (const auto byte_order = xml_attribute(*vtk_file_tag, "byte_order");
        byte_order.has_value() && *byte_order != "LittleEndian") {
      throw std::runtime_error(
          "bounded VTU loading requires byte_order=LittleEndian");
    }
    const auto header_type =
        xml_attribute(*vtk_file_tag, "header_type").value_or("UInt32");
    const std::size_t header_width =
        header_type == "UInt64" ? 8U : header_type == "UInt32" ? 4U : 0U;
    if (header_width == 0) {
      throw std::runtime_error("VTU header_type must be UInt32 or UInt64");
    }
    if (xml_attribute(*points_data_tag, "type") != "Float32" ||
        xml_attribute(*points_data_tag, "format") != "binary" ||
        parse_unsigned_attribute(*points_data_tag, "NumberOfComponents") !=
            3U) {
      throw std::runtime_error(
          "bounded VTU loading requires inline binary Float32 "
          "three-component Points");
    }

    const auto available =
        parse_unsigned_attribute(*piece_tag, "NumberOfPoints");
    if (available == 0 ||
        available > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) ||
        available > std::numeric_limits<std::uint64_t>::max() /
                        (3U * sizeof(float))) {
      throw std::runtime_error(
          "VTU NumberOfPoints is outside the supported range");
    }

    // Validate the whole grid before exposing the first piece as a point stream.
    // The tag reader skips base64 payloads in fixed-size blocks without decoding.
    const auto points_position = stream_.tellg();
    while (true) {
      const auto tag = next_xml_tag(stream_);
      if (tag.starts_with("<Piece")) {
        throw std::runtime_error(
            "bounded VTU loading supports exactly one Piece");
      }
      if (tag.starts_with("</UnstructuredGrid")) {
        break;
      }
    }
    stream_.seekg(points_position);
    total_points_ = static_cast<std::int64_t>(available);
    decoder_ = std::make_unique<Base64StreamDecoder>(stream_);
    std::array<std::uint8_t, 8> header{};
    decoder_->read_exact(header.data(), header_width);
    const auto payload_bytes =
        little_endian_unsigned(header.data(), header_width);
    if (payload_bytes != available * 3U * sizeof(float)) {
      throw std::runtime_error(
          "VTU Points payload size disagrees with NumberOfPoints");
    }
  }

  std::int64_t total_points() const { return total_points_; }
  std::int64_t point_offset() const { return point_offset_; }

  torch::Tensor read_next(const std::int64_t point_count,
                          const torch::Device& device) {
    if (point_count <= 0 || point_count > total_points_ - point_offset_) {
      throw std::invalid_argument(
          "VTU stream point count exceeds the remaining payload");
    }
    auto cpu = torch::empty(
        {point_count, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    const auto requested_bytes =
        static_cast<std::uint64_t>(point_count) * 3U * sizeof(float);
    if (requested_bytes > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("requested VTU point block is too large");
    }
    decoder_->read_exact(
        reinterpret_cast<std::uint8_t*>(cpu.data_ptr<float>()),
        static_cast<std::size_t>(requested_bytes));
    point_offset_ += point_count;
    return cpu.to(device);
  }

 private:
  std::ifstream stream_;
  std::unique_ptr<Base64StreamDecoder> decoder_;
  std::int64_t total_points_{0};
  std::int64_t point_offset_{0};
};

InlineBinaryVtuPointStream::InlineBinaryVtuPointStream(
    const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

InlineBinaryVtuPointStream::~InlineBinaryVtuPointStream() = default;
InlineBinaryVtuPointStream::InlineBinaryVtuPointStream(
    InlineBinaryVtuPointStream&&) noexcept = default;
InlineBinaryVtuPointStream& InlineBinaryVtuPointStream::operator=(
    InlineBinaryVtuPointStream&&) noexcept = default;

std::int64_t InlineBinaryVtuPointStream::total_points() const {
  return impl_->total_points();
}

std::int64_t InlineBinaryVtuPointStream::point_offset() const {
  return impl_->point_offset();
}

torch::Tensor InlineBinaryVtuPointStream::read_next(
    const std::int64_t point_count, const torch::Device& device) {
  return impl_->read_next(point_count, device);
}

Domain parse_domain(const std::string_view value) {
  if (value == "surface") {
    return Domain::kSurface;
  }
  if (value == "volume") {
    return Domain::kVolume;
  }
  throw std::invalid_argument("domain must be surface or volume");
}

std::string_view to_string(const Domain domain) {
  return domain == Domain::kSurface ? "surface" : "volume";
}

std::int64_t embedding_width(const Domain domain) {
  return domain == Domain::kSurface ? 6 : 7;
}

std::int64_t output_width(const Domain domain) {
  return domain == Domain::kSurface ? 4 : 5;
}

torch::Tensor read_inline_binary_vtu_points(
    const std::filesystem::path& path,
    const std::int64_t point_limit,
    const torch::Device& device) {
  if (point_limit < 0) {
    throw std::invalid_argument("point_limit cannot be negative");
  }
  InlineBinaryVtuPointStream stream(path);
  const auto count = resolved_count(
      stream.total_points(), point_limit, "volume");
  return stream.read_next(count, device);
}

PreparedInputs prepare_inputs(
    const std::filesystem::path& mesh_path,
    const std::filesystem::path& stl_path,
    const Domain domain,
    const torch::Device& device,
    const float air_density,
    const float stream_velocity,
    const std::int64_t point_limit) {
  auto stl = load_stl_geometry(stl_path, device);
  auto center_of_mass = stl.centers.mean(0).unsqueeze(0);
  const auto reference_scale =
      torch::tensor(
          {12.0F, 4.5F, 3.25F},
          torch::TensorOptions().dtype(torch::kFloat32).device(device))
          .reshape({1, 3});

  torch::Tensor embedding;
  float volume_length_scale = 0.0F;
  std::string mesh_reader;
  if (domain == Domain::kSurface) {
    auto surface = load_surface_features(mesh_path, point_limit, device);
    auto positions = surface.slice(-1, 0, 3);
    positions.sub_(center_of_mass);
    positions = positions / reference_scale;
    embedding = torch::cat({positions, surface.slice(-1, 3, 6)}, -1);
    mesh_reader = "vtk-xml-polydata";
  } else {
    auto positions =
        read_inline_binary_vtu_points(mesh_path, point_limit, device);
    const auto maxima = std::get<0>(stl.coordinates.max(0));
    const auto minima = std::get<0>(stl.coordinates.min(0));
    volume_length_scale = (maxima - minima).max().item<float>();

    positions.sub_(center_of_mass);
    positions = positions / reference_scale;
    auto scaled_coordinates =
        (stl.coordinates - center_of_mass) / reference_scale;
    auto sdf = warp_signed_distance(
        scaled_coordinates, stl.triangle_indices, positions);
    auto normals = positions - sdf.closest_point;
    const auto distance_to_closest = vector_norm(normals).squeeze(-1);
    const auto null_points = distance_to_closest.lt(1.0e-6);
    if (null_points.any().item<bool>()) {
      normals.index_put_(
          {null_points},
          positions.index({null_points}) - center_of_mass);
    }
    normals = normals / (vector_norm(normals) + 1.0e-6);
    embedding = torch::cat(
        {positions, sdf.distance.reshape({-1, 1}), normals}, -1);
    mesh_reader = "inline-binary-vtu-points";
  }

  auto global_features = torch::tensor(
      {air_density, stream_velocity},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto fx = global_features.reshape({1, 2})
                .expand({embedding.size(0), 2})
                .unsqueeze(0);
  return {
      fx,
      embedding.unsqueeze(0),
      volume_length_scale,
      std::move(mesh_reader),
  };
}

NormalizationStats load_normalization_stats(
    const std::filesystem::path& path,
    const Domain domain,
    const torch::Device& device) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::invalid_argument(
        "cannot open normalization statistics: " + path.string());
  }
  Json document;
  stream >> document;
  if (!document.contains("mean")) {
    throw std::invalid_argument("normalization statistics are missing mean");
  }
  const Json* deviation = nullptr;
  if (document.contains("std_dev")) {
    deviation = &document.at("std_dev");
  } else if (document.contains("std")) {
    deviation = &document.at("std");
  } else {
    throw std::invalid_argument(
        "normalization statistics are missing std_dev or std");
  }

  std::vector<float> mean;
  std::vector<float> standard_deviation;
  if (domain == Domain::kSurface) {
    append_prefix(mean, document.at("mean"), "pressure", 1);
    append_prefix(mean, document.at("mean"), "shear_stress", 3);
    append_prefix(standard_deviation, *deviation, "pressure", 1);
    append_prefix(standard_deviation, *deviation, "shear_stress", 3);
  } else {
    append_prefix(mean, document.at("mean"), "velocity", 3);
    const std::string pressure_key =
        document.at("mean").contains("pressure") ? "pressure"
                                                  : "pressure_volume";
    append_prefix(mean, document.at("mean"), pressure_key, 1);
    append_prefix(mean, document.at("mean"), "turbulent_viscosity", 1);
    append_prefix(standard_deviation, *deviation, "velocity", 3);
    append_prefix(standard_deviation, *deviation, pressure_key, 1);
    append_prefix(
        standard_deviation, *deviation, "turbulent_viscosity", 1);
  }

  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto mean_tensor = torch::from_blob(
                         mean.data(),
                         {static_cast<std::int64_t>(mean.size())},
                         options)
                         .clone()
                         .to(device);
  auto deviation_tensor =
      torch::from_blob(
          standard_deviation.data(),
          {static_cast<std::int64_t>(standard_deviation.size())},
          options)
          .clone()
          .to(device);
  return {std::move(mean_tensor), std::move(deviation_tensor)};
}

torch::Tensor decode_physical_outputs(
    const torch::Tensor& standardized,
    const NormalizationStats& stats,
    const Domain domain,
    const double air_density,
    const double stream_velocity,
    const double volume_length_scale) {
  if (standardized.size(-1) != output_width(domain)) {
    throw std::invalid_argument("standardized output has the wrong channel count");
  }
  auto unscaled = standardized * stats.standard_deviation + stats.mean;
  const double dynamic_pressure =
      air_density * stream_velocity * stream_velocity;
  if (domain == Domain::kSurface) {
    auto pressure = unscaled.slice(-1, 0, 1) * dynamic_pressure;
    auto wall_shear_stress = unscaled.slice(-1, 1, 4) * dynamic_pressure;
    return torch::cat({pressure, wall_shear_stress}, -1);
  }
  auto velocity = unscaled.slice(-1, 0, 3) * stream_velocity;
  auto pressure = unscaled.slice(-1, 3, 4) * dynamic_pressure;
  const double turbulent_viscosity_scale =
      stream_velocity * volume_length_scale;
  auto turbulent_viscosity =
      unscaled.slice(-1, 4, 5) * turbulent_viscosity_scale;
  return torch::cat({velocity, pressure, turbulent_viscosity}, -1);
}

void validate_package_contract(
    const physicsnemo::inference::ModelManifest& manifest,
    const Domain domain,
    const std::int64_t block_size) {
  if (block_size < 2) {
    throw std::invalid_argument("block_size must be at least 2");
  }
  if (manifest.inputs.size() != 2U || manifest.outputs.size() != 1U) {
    throw std::invalid_argument(
        "Transolver package must contain two inputs and one output");
  }
  validate_tensor(
      tensor_spec(manifest.inputs, "fx"), 2, block_size, "fx");
  validate_tensor(
      tensor_spec(manifest.inputs, "embedding"),
      embedding_width(domain),
      block_size,
      "embedding");
  validate_tensor(
      manifest.outputs.front(),
      output_width(domain),
      block_size,
      "output");
}

}  // namespace transolver_workflow
