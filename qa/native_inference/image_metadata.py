"""Record the source and matched native/Python VTK identities in the QA image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def validate_identity(
    source_sha: str,
    native_vtk: str,
    python_vtk: str,
    python_vtk_path: Path,
    distro_python_root: Path,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("SOURCE_SHA must be the full 40-character Git SHA")
    if native_vtk != python_vtk:
        raise ValueError("native and Python VTK versions must match")
    if not python_vtk_path.resolve().is_relative_to(distro_python_root.resolve()):
        raise ValueError("Python VTK must use the matching distro package")
    return {"format_version": 1, "source_sha": source_sha, "vtk_version": python_vtk}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--cuda-architectures", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tensorrt
    import torch
    from vtkmodules import vtkCommonCore

    headers = list(Path("/usr/include").glob("vtk-*/vtkVersionMacros.h"))
    if len(headers) != 1:
        raise ValueError("expected one distro VTK development header")
    match = re.search(r'#define\s+VTK_VERSION\s+"([^"]+)"', headers[0].read_text())
    if match is None:
        raise ValueError("cannot read the native VTK version")
    identity = validate_identity(
        args.source_sha,
        match[1],
        vtkCommonCore.vtkVersion.GetVTKVersion(),
        Path(vtkCommonCore.__file__),
        Path("/usr/lib/python3/dist-packages"),
    )
    identity.update(
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        tensorrt_version=tensorrt.__version__,
        cuda_architectures=args.cuda_architectures,
    )
    args.output.write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
