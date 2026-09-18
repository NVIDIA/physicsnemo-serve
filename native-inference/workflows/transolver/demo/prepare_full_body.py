#!/usr/bin/env python3
"""Prepare a complete, coarsened vehicle surface for the bounded live demo.

This runs offline. The generated VTP is geometry only, not a reduced CFD result.
Every output triangle is subsequently predicted by the native workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source: Path, output: Path, provenance: Path, cells: int) -> dict:
    import vtk

    if cells < 4 or cells % 2:
        raise ValueError("cells must be an even integer of at least four")
    if output.exists() or provenance.exists():
        raise FileExistsError("output and provenance must be new files")
    started = time.monotonic()
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(source))
    reader.Update()
    mesh = reader.GetOutput()
    source_count = mesh.GetNumberOfCells()
    if source_count <= cells or mesh.GetNumberOfPolys() != source_count:
        raise ValueError("source must contain more surface triangles than cells")
    source_bounds = list(mesh.GetBounds())
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(1.0 - cells / source_count)
    decimate.AttributeErrorMetricOff()
    decimate.VolumePreservationOn()
    decimate.Update()
    result = decimate.GetOutput()
    output_count = result.GetNumberOfCells()
    if output_count != cells:
        raise ValueError(f"decimator produced {output_count} cells, expected {cells}")
    output_bounds = list(result.GetBounds())
    # Detect a partial surface or a large loss of vehicle extent before export.
    for axis in range(3):
        span = source_bounds[axis * 2 + 1] - source_bounds[axis * 2]
        for end in range(2):
            i = axis * 2 + end
            if abs(output_bounds[i] - source_bounds[i]) > span * 0.02:
                raise ValueError("coarsened surface lost more than 2% of an axis extent")
    result.GetPointData().Initialize()
    result.GetCellData().Initialize()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(result)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError("could not write coarsened surface")
    receipt = {
        "schema_version": 1,
        "label": "Coarsened demonstration mesh",
        "method": "VTK quadric decimation of complete vehicle STL; volume preservation enabled",
        "source_mesh": str(source.resolve()),
        "source_mesh_sha256": sha256(source),
        "source_cell_count": source_count,
        "source_bounds": source_bounds,
        "mesh": str(output.resolve()),
        "mesh_sha256": sha256(output),
        "output_cell_count": output_count,
        "output_point_count": result.GetNumberOfPoints(),
        "output_bounds": output_bounds,
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "preparation_seconds": time.monotonic() - started,
        "coverage": "Complete vehicle surface; every output cell receives a native prediction",
        "limitations": "Geometry-only coarsening for demonstration. No CFD ground truth is supplied or implied.",
    }
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--cells", default=65536, type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.stl, args.output, args.provenance, args.cells), indent=2))


if __name__ == "__main__":
    main()
