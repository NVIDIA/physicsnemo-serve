"""Independent eager reference for the documented raw Transolver surface flow.

This module reads the original geometry and trusted PhysicsNeMo checkpoint. It
does not import Model Builder, consume its converted weights, or use native input
dumps as reference inputs. The geometry contract is the surface workflow adapted
from gpu_programming 87b78bf6cec3030cfbaea10f14e4c64cbc836407, as documented in
native-inference/workflows/transolver/README.md. Matching Python/C++ VTK versions
are required because cell centers and normals can change between VTK releases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def block_plan(point_count: int, block_size: int) -> list[int]:
    if point_count < 2 or block_size < 2:
        raise ValueError("point_count and block_size must be at least two")
    if point_count % block_size == 1:
        raise ValueError("the final one-point block is outside the qualified contract")
    return [
        min(block_size, point_count - start)
        for start in range(0, point_count, block_size)
    ]


def surface_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(Path(path).read_text())
    deviation = document.get("std_dev", document.get("std"))
    if not isinstance(document.get("mean"), dict) or not isinstance(deviation, dict):
        raise ValueError("surface statistics require mean and std_dev or std")

    def channels(values):
        pressure, shear = values.get("pressure"), values.get("shear_stress")
        if not isinstance(pressure, list) or len(pressure) < 1:
            raise ValueError("surface pressure statistics require one channel")
        if not isinstance(shear, list) or len(shear) < 3:
            raise ValueError("surface shear_stress statistics require three channels")
        result = np.asarray([*pressure[:1], *shear[:3]], dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("surface statistics contain nonfinite values")
        return result

    mean, std = channels(document["mean"]), channels(deviation)
    if np.any(std <= 0):
        raise ValueError("surface standard deviations must be positive")
    return mean, std


def decode_surface(standardized, mean, std, density=1.205, velocity=30.0):
    """Independent FP32 decoding; accepts NumPy arrays or eager Torch tensors."""
    if not math.isfinite(density) or not math.isfinite(velocity):
        raise ValueError("flow conditions must be finite")
    if standardized.shape[-1] != 4:
        raise ValueError("surface output must have four channels")
    return (standardized * std + mean) * (density * velocity * velocity)


def configure_determinism():
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublas")


def load_original_model(checkpoint: Path, device="cuda:0"):
    import physicsnemo
    import torch
    from physicsnemo.models.transolver import Transolver

    with torch.device("cpu"):
        model = physicsnemo.Module.from_checkpoint(str(checkpoint), strict=True)
    if not isinstance(model, Transolver):
        raise ValueError("reference checkpoint must contain an upstream Transolver")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise ValueError("reference checkpoint must preserve FP32 model state")
    return model.to(device).eval()


def prepare_surface(
    vtp: Path,
    stl: Path,
    point_count: int,
    *,
    device="cuda:0",
    density=1.205,
    velocity=30.0,
):
    import torch
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    def centers(mesh):
        operation = vtk.vtkCellCenters()
        operation.SetInputData(mesh)
        operation.VertexCellsOff()
        operation.Update()
        points = operation.GetOutput().GetPoints()
        if points is None or points.GetNumberOfPoints() != mesh.GetNumberOfCells():
            raise ValueError("geometry has missing or invalid cell centers")
        return np.array(vtk_to_numpy(points.GetData()), dtype=np.float32, copy=True)

    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(str(stl))
    stl_reader.MergingOn()
    stl_reader.Update()
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(stl_reader.GetOutputPort())
    triangulate.Update()
    stl_mesh = triangulate.GetOutput()
    if not stl_mesh.GetNumberOfPoints() or not stl_mesh.GetNumberOfPolys():
        raise ValueError("STL geometry is empty")
    stl_centers = torch.from_numpy(centers(stl_mesh)).to(device)
    center_of_mass = stl_centers.mean(dim=0).unsqueeze(0)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp))
    reader.Update()
    mesh = reader.GetOutput()
    normal_filter = None
    if mesh.GetCellData().GetNormals() is None:
        normal_filter = vtk.vtkPolyDataNormals()
        normal_filter.SetInputData(mesh)
        normal_filter.ComputeCellNormalsOn()
        normal_filter.ComputePointNormalsOff()
        normal_filter.SplittingOff()
        normal_filter.ConsistencyOn()
        normal_filter.AutoOrientNormalsOff()
        normal_filter.FlipNormalsOff()
        normal_filter.NonManifoldTraversalOn()
        normal_filter.Update()
        mesh = normal_filter.GetOutput()
    if point_count < 2 or point_count > mesh.GetNumberOfCells():
        raise ValueError("point count is outside the surface cell range")
    positions = torch.from_numpy(centers(mesh)[:point_count].copy()).to(device)
    raw_normals = mesh.GetCellData().GetNormals()
    if raw_normals is None or raw_normals.GetNumberOfComponents() != 3:
        raise ValueError("surface cell normals are missing")
    normals = np.array(
        vtk_to_numpy(raw_normals)[:point_count], dtype=np.float32, copy=True
    )
    if normals.shape != (point_count, 3):
        raise ValueError("surface cell normals are truncated")
    # Preserve the documented FP32 CPU normalization before the GPU unit norm.
    squared = normals * normals
    lengths = np.sqrt((squared[:, 0] + squared[:, 1]) + squared[:, 2]) + np.float32(
        1e-8
    )
    normals = torch.from_numpy(normals / lengths[:, None]).to(device)
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    scale = torch.tensor([12.0, 4.5, 3.25], dtype=torch.float32, device=device)
    embedding = torch.cat(
        ((positions - center_of_mass) / scale, normals), dim=-1
    ).unsqueeze(0)
    fx = torch.tensor([density, velocity], dtype=torch.float32, device=device)
    fx = fx.reshape(1, 1, 2).expand(1, point_count, 2)
    if not torch.isfinite(embedding).all() or not torch.isfinite(fx).all():
        raise ValueError("reference geometry produced nonfinite inputs")
    return fx, embedding, vtk.vtkVersion.GetVTKVersion()


def eager_surface(
    model,
    fx,
    embedding,
    mean,
    std,
    *,
    block_size=75,
    seed=0,
    density=1.205,
    velocity=30.0,
):
    """Run eager batches and independently scatter them back to source order."""
    import torch

    count = embedding.shape[1]
    plan = block_plan(count, block_size)
    if fx.shape != (1, count, 2) or embedding.shape != (1, count, 6):
        raise ValueError("surface reference inputs have incompatible shapes")
    for value in (fx, embedding):
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("surface reference inputs must be finite FP32")
    mean = torch.as_tensor(mean, dtype=torch.float32, device=fx.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=fx.device)
    torch.manual_seed(seed)
    permutation = torch.randperm(count, device=fx.device, dtype=torch.int64)
    standardized = torch.empty((count, 4), dtype=torch.float32)
    physical = torch.empty_like(standardized)
    with torch.inference_mode():
        begin = 0
        for points in plan:
            indices = permutation[begin : begin + points]
            prediction = model(
                fx.index_select(1, indices).contiguous(),
                embedding.index_select(1, indices).contiguous(),
            )
            if prediction.shape != (1, points, 4) or prediction.dtype != torch.float32:
                raise ValueError(
                    "upstream model returned an incompatible surface output"
                )
            if not torch.isfinite(prediction).all():
                raise ValueError("upstream model returned nonfinite values")
            decoded = decode_surface(prediction, mean, std, density, velocity)
            standardized.index_copy_(0, indices.cpu(), prediction.squeeze(0).cpu())
            physical.index_copy_(0, indices.cpu(), decoded.squeeze(0).cpu())
            begin += points
    return {
        "fx": fx.detach().cpu().contiguous().numpy(),
        "embedding": embedding.detach().cpu().contiguous().numpy(),
        "standardized_output": standardized.unsqueeze(0).numpy(),
        "physical_output": physical.unsqueeze(0).numpy(),
    }
