"""Verify source-cell mapping and that validation badges bind to current bytes."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import vtk
except ImportError:
    vtk = None


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "export_viewer.py"
SPEC = importlib.util.spec_from_file_location("export_viewer", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ViewerExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "inputs").mkdir()
        self.paths = {
            "physical_output": self.root / "physical.f32",
            "standardized_output": self.root / "standardized.f32",
            "fx": self.root / "inputs" / "fx.f32",
            "embedding": self.root / "inputs" / "embedding.f32",
        }
        self.write_tensor("physical_output", [-200, 3, 4, 0, 100, 0, 0, 2])
        self.write_tensor("standardized_output", list(range(8)))
        self.write_tensor("fx", [1.205, 30, 1.205, 30])
        self.write_tensor("embedding", list(range(12)))
        self.metadata = {
            "domain": "surface", "output_dtype": "float32",
            "output_shape": [1, 2, 4], "point_count": 2, "point_limit": 2,
            "block_size": 2, "block_count": 1, "permutation_seed": 0,
            "mesh": "case/boundary.vtp", "stl": "case/car.stl",
            "air_density": 1.205, "stream_velocity": 30.0,
            "backend": "aoti", "preparation_ms": 12, "inference_ms": 4,
            "physical_output": str(self.paths["physical_output"]),
            "standardized_output": str(self.paths["standardized_output"]),
        }
        self.metadata_path = self.root / "metadata.json"
        self.save_metadata()
        self.geometry = {
            "source_cell_count": 10,
            "context_mesh": {"vertices": [[0, 0, 0]], "triangles": []},
            "prediction_mesh": {
                "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                "triangles": [[0, 1, 2], [0, 2, 3], [0, 1, 3]],
                "triangle_cell_ids": [0, 0, 1], "source_cell_ids": [0, 1],
            },
            "centers": [[0.5, 0.5, 0], [0.3, 0.3, 0]],
        }
        self.report = {
            "native_metadata": dict(self.metadata),
            "comparisons": {
                name: {"reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "bitwise_equal": True}
                for name, path in self.paths.items()
            },
        }
        self.report_path = self.root / "reference.json"
        self.save_report()

    def write_tensor(self, name, values):
        self.paths[name].write_bytes(struct.pack("<" + "f" * len(values), *values))

    def save_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata))

    def save_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def payload(self, reference=False, mesh_provenance=None):
        return EXPORT.build_payload(
            self.metadata_path, self.report_path if reference else None,
            mesh_provenance=mesh_provenance,
            geometry_loader=lambda metadata, count: self.geometry,
        )

    def test_prediction_rows_are_source_cells_and_flat_triangles(self):
        result = self.payload()
        self.assertEqual(result["fields"], [[-200, 3, 4, 0], [100, 0, 0, 2]])
        self.assertEqual(result["prediction_mesh"]["source_cell_ids"], [0, 1])
        face_pressure = [result["fields"][i][0]
                         for i in result["prediction_mesh"]["triangle_cell_ids"]]
        self.assertEqual(face_pressure, [-200, -200, 100])
        self.assertNotIn("fields", result["context_mesh"])
        self.assertEqual(result["validation"]["status"], "not_compared")

    def test_saved_reference_badge_requires_all_current_tensors(self):
        result = self.payload(reference=True)
        self.assertEqual(result["validation"]["status"], "matched")
        self.assertEqual(result["validation"]["label"], "Matches saved Python reference")
        self.assertEqual(set(result["validation"]["comparisons"]), set(self.paths))

    def test_changed_current_bytes_fail_even_when_report_says_equal(self):
        for tensor in self.paths:
            with self.subTest(tensor=tensor):
                original = self.paths[tensor].read_bytes()
                self.paths[tensor].write_bytes(struct.pack("<f", 99) + original[4:])
                with self.assertRaisesRegex(ValueError, tensor):
                    self.payload(reference=True)
                self.paths[tensor].write_bytes(original)

    def test_reference_with_different_source_or_settings_fails(self):
        for key, value in [("mesh", "other.vtp"), ("stl", "other.stl"),
                           ("permutation_seed", 1), ("block_size", 3),
                           ("air_density", 1.0), ("stream_velocity", 40),
                           ("point_limit", 0)]:
            with self.subTest(key=key):
                original = self.report["native_metadata"][key]
                self.report["native_metadata"][key] = value
                self.save_report()
                with self.assertRaisesRegex(ValueError, key):
                    self.payload(reference=True)
                self.report["native_metadata"][key] = original

    def test_nonfinite_and_wrong_length_are_rejected(self):
        for tensor in ("physical_output", "standardized_output"):
            original = self.paths[tensor].read_bytes()
            for bad in (original[:-4], original + struct.pack("<f", 1),
                        struct.pack("<f", math.nan) + original[4:],
                        struct.pack("<f", math.inf) + original[4:]):
                with self.subTest(tensor=tensor, length=len(bad)):
                    self.paths[tensor].write_bytes(bad)
                    with self.assertRaisesRegex(ValueError, tensor):
                        self.payload()
            self.paths[tensor].write_bytes(original)

    def test_bad_metadata_cannot_silently_change_cell_association(self):
        for key, value in [("output_shape", [1, 3, 4]), ("output_dtype", "float64"),
                           ("domain", "volume"), ("point_limit", 1),
                           ("point_count", True)]:
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                self.save_metadata()
                with self.assertRaisesRegex(ValueError, key):
                    self.payload()
                self.metadata[key] = original
        self.metadata["point_limit"] = 0
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "point_limit"):
            self.payload()

    def test_missing_input_dump_cannot_receive_reference_badge(self):
        self.paths["embedding"].unlink()
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "embedding"):
            self.payload(reference=True)

    def provenance(self):
        mesh = self.root / self.metadata["mesh"]
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"coarsened complete surface geometry")
        report = {
            "label": "Coarsened demonstration mesh", "source_cell_count": 100,
            "output_cell_count": 10, "method": "VTK quadric decimation of complete vehicle STL",
            "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        }
        path = self.root / "mesh-provenance.json"
        path.write_text(json.dumps(report))
        return path, report, mesh

    def test_explicit_provenance_binds_to_actual_mesh_and_source_count(self):
        path, report, _ = self.provenance()
        result = self.payload(mesh_provenance=path)
        self.assertEqual(result.get("mesh_provenance"), report)
        self.assertNotIn("mesh_provenance", self.payload(), "Old results must not acquire invented mesh provenance")

    def test_changed_mesh_or_false_provenance_cell_count_fails(self):
        path, report, mesh = self.provenance()
        mesh.write_bytes(b"different surface geometry")
        with self.assertRaisesRegex(ValueError, "mesh_sha256"):
            self.payload(mesh_provenance=path)
        report["mesh_sha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        report["output_cell_count"] = 9
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "output_cell_count"):
            self.payload(mesh_provenance=path)


@unittest.skipIf(vtk is None, "requires Python VTK (run on the qualified H100)")
class ViewerGeometryTest(unittest.TestCase):
    def test_adjacent_quads_share_vertices_without_losing_cell_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = vtk.vtkPoints()
            for point in [(0, 0, 0), (1, 0, 0), (1, 1, 0),
                          (0, 1, 0), (2, 0, 0), (2, 1, 0)]:
                points.InsertNextPoint(*point)
            cells = vtk.vtkCellArray()
            for indices in [(0, 1, 2, 3), (1, 4, 5, 2)]:
                cell = vtk.vtkQuad()
                for index, point_id in enumerate(indices):
                    cell.GetPointIds().SetId(index, point_id)
                cells.InsertNextCell(cell)
            mesh = vtk.vtkPolyData()
            mesh.SetPoints(points)
            mesh.SetPolys(cells)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(root / "mesh.vtp"))
            writer.SetInputData(mesh)
            writer.Write()
            triangles = vtk.vtkTriangleFilter()
            triangles.SetInputData(mesh)
            stl = vtk.vtkSTLWriter()
            stl.SetFileName(str(root / "mesh.stl"))
            stl.SetInputConnection(triangles.GetOutputPort())
            stl.Write()

            result = EXPORT.load_geometry(
                {"mesh": str(root / "mesh.vtp"), "stl": str(root / "mesh.stl")}, 2)
            patch = result["prediction_mesh"]
            self.assertEqual(len(patch["vertices"]), 6,
                             "shared source vertices should appear only once")
            self.assertEqual(patch["source_cell_ids"], [0, 1])
            self.assertEqual(patch["triangle_cell_ids"], [0, 0, 1, 1])
            self.assertEqual(result["centers"], [[0.5, 0.5, 0], [1.5, 0.5, 0]])
            for cell_id in (0, 1):
                face_points = {
                    tuple(patch["vertices"][point_id])
                    for triangle, parent in zip(patch["triangles"], patch["triangle_cell_ids"])
                    if parent == cell_id for point_id in triangle
                }
                self.assertEqual(face_points, {
                    (cell_id, 0, 0), (cell_id + 1, 0, 0),
                    (cell_id, 1, 0), (cell_id + 1, 1, 0),
                })


if __name__ == "__main__":
    unittest.main()
