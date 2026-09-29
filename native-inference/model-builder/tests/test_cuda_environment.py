"""CUDA toolkit selection is testable without compiling or importing Torch."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))


class CudaEnvironmentTests(unittest.TestCase):
    def test_selects_platform_ptxas_and_preserves_an_explicit_override(self):
        torch = SimpleNamespace(
            **{
                name: name
                for name in (
                    "float32",
                    "float16",
                    "bfloat16",
                    "int32",
                    "int64",
                    "uint8",
                )
            }
        )
        for platform, executable in (("linux", "ptxas"), ("win32", "ptxas.exe")):
            with (
                self.subTest(platform=platform),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "bin").mkdir()
                assembler = root / "bin" / executable
                assembler.write_bytes(b"assembler fixture; never executed")
                spec = importlib.util.spec_from_file_location(
                    "cuda_environment_exporter_test",
                    SOURCE / "pnmir_export/exporter.py",
                )
                exporter = importlib.util.module_from_spec(spec)
                with (
                    mock.patch.dict(
                        sys.modules,
                        {
                            "torch": torch,
                            "torch.utils.cpp_extension": SimpleNamespace(
                                CUDA_HOME=str(root)
                            ),
                        },
                    ),
                    mock.patch.object(sys, "platform", platform),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    spec.loader.exec_module(exporter)
                    exporter._configure_cuda_export_environment()
                    self.assertEqual(
                        os.environ.get("TRITON_PTXAS_PATH"), str(assembler)
                    )
                    os.environ["TRITON_PTXAS_PATH"] = "customer-ptxas"
                    exporter._configure_cuda_export_environment()
                    self.assertEqual(os.environ["TRITON_PTXAS_PATH"], "customer-ptxas")


if __name__ == "__main__":
    unittest.main()
