#!/usr/bin/env python3
"""Export native surface results to a self-contained, read-only HTML viewer."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def _tensor(path, name, count, width):
    data = Path(path).read_bytes()
    expected = count * width * 4
    if len(data) != expected:
        raise ValueError(f"{name}: expected {expected} float32 bytes, got {len(data)}")
    values = [value[0] for value in struct.iter_unpack("<f", data)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name}: contains nonfinite values")
    return values, hashlib.sha256(data).hexdigest()


def _path(metadata_path, value):
    path = Path(value)
    return path if path.is_absolute() else metadata_path.parent / path


def _compare_reference(path, metadata, metadata_path, hashes, count):
    report = json.loads(Path(path).read_text())
    reference_metadata = report.get("native_metadata", report.get("metadata", {}))
    # Output destinations, timing, and package paths can change between runs.
    for key in ("domain", "output_dtype", "output_shape", "point_count",
                "point_limit", "block_size", "block_count", "permutation_seed",
                "mesh", "stl", "air_density", "stream_velocity"):
        if key not in reference_metadata or reference_metadata[key] != metadata.get(key):
            raise ValueError(f"reference metadata mismatch: {key}")
    comparisons = {}
    widths = {"fx": 2, "embedding": 6, "standardized_output": 4, "physical_output": 4}
    for name, width in widths.items():
        if name not in hashes:
            _, hashes[name] = _tensor(
                metadata_path.parent / "inputs" / f"{name}.f32", name, count, width)
        saved = report.get("comparisons", {}).get(name, {})
        expected = saved.get("reference_sha256", saved.get("python_reference_sha256"))
        if not expected or hashes[name] != expected:
            raise ValueError(f"{name}: current bytes do not match the saved Python reference")
        if saved.get("bitwise_equal") is False or saved.get("hash_equal") is False:
            raise ValueError(f"{name}: saved reference report records a mismatch")
        if (saved.get("reference_shape", [1, count, width]) != [1, count, width]
                or saved.get("reference_dtype", "float32") != "float32"):
            raise ValueError(f"{name}: saved reference shape or dtype mismatch")
        comparisons[name] = {
            "sha256": hashes[name], "python_reference_sha256": expected, "hash_equal": True,
        }
    return {
        "status": "matched", "label": "Matches saved Python reference",
        "reference_report": str(Path(path).resolve()), "comparisons": comparisons,
    }


def load_geometry(metadata, count):
    """Keep native cell order and use STL decimation only for gray context."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    for name in ("mesh", "stl"):
        if not Path(metadata[name]).is_file():
            raise ValueError(f"{name}: file does not exist: {metadata[name]}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(metadata["mesh"])
    reader.Update()
    mesh = reader.GetOutput()
    # Match load_surface_features in workflow.cpp before selecting cell indices.
    if mesh.GetCellData().GetNormals() is None:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.FlipNormalsOff()
        normals.NonManifoldTraversalOn()
        normals.Update()
        mesh = normals.GetOutput()
    source_count = mesh.GetNumberOfCells()
    if count > source_count:
        raise ValueError("point_count exceeds source mesh cell count")
    centers_filter = vtk.vtkCellCenters()
    centers_filter.SetInputData(mesh)
    centers_filter.VertexCellsOff()
    centers_filter.Update()
    if centers_filter.GetOutput().GetNumberOfPoints() != source_count:
        raise ValueError("source mesh contains cells without a valid cell center")
    centers = vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData())[:count]
    vertices, triangles, cell_ids = [], [], []
    vertex_indices = {}
    ids, points = vtk.vtkIdList(), vtk.vtkPoints()
    for source_id in range(count):
        cell = mesh.GetCell(source_id)
        if cell.GetCellDimension() != 2:
            raise ValueError(f"source cell {source_id} is not a surface cell")
        if not cell.Triangulate(0, ids, points) or points.GetNumberOfPoints() % 3:
            raise ValueError(f"cannot triangulate source cell {source_id}")
        for offset in range(0, points.GetNumberOfPoints(), 3):
            triangle = []
            for index in range(offset, offset + 3):
                point_id = ids.GetId(index)
                if point_id not in vertex_indices:
                    vertex_indices[point_id] = len(vertices)
                    vertices.append(list(points.GetPoint(index)))
                triangle.append(vertex_indices[point_id])
            triangles.append(triangle)
            cell_ids.append(source_id)
    if not triangles:
        raise ValueError("source selection has no surface triangles")

    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(metadata["stl"])
    stl_reader.Update()
    context = stl_reader.GetOutput()
    context_count = context.GetNumberOfCells()
    if not context_count:
        raise ValueError("stl: no surface triangles found")
    if context_count > 10000:
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(context)
        decimator.SetTargetReduction(1 - 10000 / context_count)
        decimator.Update()
        context = decimator.GetOutput()
    context_vertices = vtk_to_numpy(context.GetPoints().GetData())
    context_faces = vtk_to_numpy(context.GetPolys().GetData()).reshape(-1, 4)
    if not np.all(context_faces[:, 0] == 3):
        raise ValueError("stl: context contains nontriangle cells")
    if not (np.isfinite(context_vertices).all() and np.isfinite(vertices).all()
            and np.isfinite(centers).all()):
        raise ValueError("mesh geometry contains nonfinite coordinates")
    return {
        "source_cell_count": source_count,
        "context_mesh": {
            "vertices": context_vertices.tolist(), "triangles": context_faces[:, 1:].tolist(),
        },
        "prediction_mesh": {
            "vertices": vertices, "triangles": triangles,
            "triangle_cell_ids": cell_ids, "source_cell_ids": list(range(count)),
        },
        "centers": centers.tolist(),
    }


def build_payload(metadata_path, reference_report=None, *, mesh_provenance=None,
                  geometry_loader=None):
    metadata_path = Path(metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("domain") != "surface":
        raise ValueError("domain: this viewer supports surface results only")
    count = metadata.get("point_count")
    if type(count) is not int or count <= 0:
        raise ValueError("point_count must be a positive integer")
    if metadata.get("output_dtype") != "float32":
        raise ValueError("output_dtype must be float32")
    if metadata.get("output_shape") != [1, count, 4]:
        raise ValueError("output_shape must be [1, point_count, 4]")
    limit = metadata.get("point_limit")
    if type(limit) is not int or limit not in (0, count):
        raise ValueError("point_limit must be 0 or equal point_count (source prefix)")
    values, physical_hash = _tensor(
        _path(metadata_path, metadata["physical_output"]), "physical_output", count, 4)
    _, standardized_hash = _tensor(
        _path(metadata_path, metadata["standardized_output"]), "standardized_output", count, 4)
    hashes = {"physical_output": physical_hash, "standardized_output": standardized_hash}
    validation = {
        "status": "not_compared", "label": "Python reference not compared",
        "reference_report": None, "comparisons": {},
    }
    if reference_report:
        validation = _compare_reference(
            reference_report, metadata, metadata_path, hashes, count)
    geometry_metadata = dict(metadata)
    for name in ("mesh", "stl"):
        geometry_metadata[name] = str(_path(metadata_path, metadata[name]))
    geometry = (geometry_loader or load_geometry)(geometry_metadata, count)
    source_count = geometry["source_cell_count"]
    if count > source_count:
        raise ValueError("point_count exceeds source mesh cell count")
    if limit == 0 and count != source_count:
        raise ValueError("point_limit=0 requires all source cells")
    payload = {
        "metadata": metadata,
        "summary": {
            "point_count": count, "source_cell_count": source_count,
            "mesh_name": Path(metadata["mesh"]).name, "backend": metadata["backend"],
            "preparation_ms": metadata["preparation_ms"],
            "inference_ms": metadata["inference_ms"],
        },
        "fields": [values[i:i + 4] for i in range(0, len(values), 4)],
        "context_mesh": geometry["context_mesh"],
        "prediction_mesh": geometry["prediction_mesh"], "centers": geometry["centers"],
        "validation": validation, "hashes": hashes,
    }
    if mesh_provenance:
        provenance = json.loads(Path(mesh_provenance).read_text())
        mesh_hash = hashlib.sha256(Path(geometry_metadata["mesh"]).read_bytes()).hexdigest()
        if provenance.get("mesh_sha256") != mesh_hash:
            raise ValueError("mesh provenance mesh_sha256 does not match the current mesh")
        if (type(provenance.get("output_cell_count")) is not int
                or provenance["output_cell_count"] != source_count):
            raise ValueError("mesh provenance output_cell_count does not match the current mesh")
        if (type(provenance.get("source_cell_count")) is not int
                or provenance["source_cell_count"] <= 0):
            raise ValueError("mesh provenance source_cell_count must be positive")
        for key in ("label", "method"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise ValueError(f"mesh provenance {key} must be a nonempty string")
        payload["mesh_provenance"] = provenance
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--mesh-provenance", type=Path,
                        help="Optional provenance JSON bound to this mesh's SHA-256 and cell count")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = build_payload(args.metadata, args.reference_report,
                                mesh_provenance=args.mesh_provenance)
        from plotly.offline import get_plotlyjs

        template = Path(__file__).with_name("viewer.html").read_text()
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        html = template.replace("__PLOTLY_JS__", get_plotlyjs()).replace("__RESULT_DATA__", data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"Viewer export failed: {error}\n")
    print(f"Exported {args.output.resolve()} — {payload['validation']['label']}")
    print(f"{payload['summary']['point_count']} predicted cells; "
          f"{len(payload['context_mesh']['triangles'])} context triangles; "
          f"{args.output.stat().st_size:,} bytes (self-contained HTML)")


if __name__ == "__main__":
    main()
