"""Compile the required scatter support extension without exposing a GPU.

The Transolver inference path does not call this extension; the unchanged CFD
plugin requires it for readiness. Numerical runtime dependencies are inherited.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

SOURCE_URL = "https://files.pythonhosted.org/packages/f5/ab/2a44ecac0f891dd0d765fc59ac8d277c6283a31907626560e72685df2ed6/torch_scatter-2.1.2.tar.gz"
SOURCE_SHA256 = "69b3aa435f2424ac6a1bfb6ff702da6eb73b33ca0db38fb26989c74159258e47"
OUTPUT = Path("/opt/python-cfd-rest-scatter")


def main():
    import torch

    OUTPUT.mkdir(exist_ok=False)
    archive = OUTPUT / "torch_scatter-2.1.2.tar.gz"
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        archive.write_bytes(response.read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SOURCE_SHA256
    with tarfile.open(archive) as source:
        source.extractall(OUTPUT, filter="data")
    environment = dict(
        os.environ,
        NVIDIA_VISIBLE_DEVICES="void",
        CUDA_VISIBLE_DEVICES="",
        FORCE_CUDA="1",
        TORCH_CUDA_ARCH_LIST="9.0",
        MAX_JOBS="8",
        PIP_NO_INDEX="1",
    )
    subprocess.run(
        ["python3", "setup.py", "bdist_wheel", "--dist-dir", str(OUTPUT)],
        cwd=OUTPUT / "torch_scatter-2.1.2",
        env=environment,
        check=True,
    )
    (wheel,) = OUTPUT.glob("*.whl")
    receipt = {
        "schema_version": 1,
        "gpu_execution": False,
        "source_version": "2.1.2",
        "production_distribution": "torch-scatter==2.1.2+pt210cu128",
        "production_source_commit": None,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "cuda_arch_list": "9.0",
        "force_cuda": True,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "cxx": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "wheel": {
            "filename": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "size_bytes": wheel.stat().st_size,
        },
    }
    (OUTPUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
