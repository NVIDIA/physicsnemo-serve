"""Geometry and provenance checks for the offline full-body demo preparation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import vtk
except ImportError:
    vtk = None

MODULE_PATH = Path(__file__).resolve().parents[1] / "demo" / "prepare_full_body.py"
SPEC = importlib.util.spec_from_file_location("prepare_full_body", MODULE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@unittest.skipIf(vtk is None, "VTK is required for mesh preparation")
class PrepareFullBodyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "whole-body.stl"
        self.output = self.directory / "coarse.vtp"
        self.provenance = self.directory / "provenance.json"
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(34)
        sphere.SetPhiResolution(34)
        sphere.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(self.source))
        writer.SetInputData(sphere.GetOutput())
        writer.SetFileTypeToBinary()
        self.assertEqual(writer.Write(), 1)

    def test_complete_surface_and_provenance_binding(self):
        receipt = PREPARE.prepare(self.source, self.output, self.provenance, 1024)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(self.output))
        reader.Update()
        mesh = reader.GetOutput()
        self.assertEqual(mesh.GetNumberOfCells(), 1024)
        self.assertEqual(mesh.GetNumberOfPolys(), 1024)
        self.assertEqual(mesh.GetCellData().GetNumberOfArrays(), 0)
        self.assertEqual(mesh.GetPointData().GetNumberOfArrays(), 0)
        bounds = mesh.GetBounds()
        for axis in range(3):
            self.assertLess(bounds[2 * axis], -0.48)
            self.assertGreater(bounds[2 * axis + 1], 0.48)
        self.assertEqual(receipt["mesh_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(receipt["source_mesh_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(receipt["output_cell_count"], 1024)
        self.assertEqual(json.loads(self.provenance.read_text()), receipt)

    def test_existing_output_is_preserved(self):
        self.output.write_bytes(b"existing recording input")
        with self.assertRaises(FileExistsError):
            PREPARE.prepare(self.source, self.output, self.provenance, 1024)
        self.assertEqual(self.output.read_bytes(), b"existing recording input")
        self.assertFalse(self.provenance.exists())

    def test_invalid_target_does_not_create_outputs(self):
        for target in (0, 3, 1025, 10000):
            with self.assertRaises(ValueError):
                PREPARE.prepare(self.source, self.output, self.provenance, target)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.provenance.exists())


if __name__ == "__main__":
    unittest.main()
