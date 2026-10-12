"""AOTI export precision boundaries and restoration; compilation is stubbed."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class ExportPrecisionTests(unittest.TestCase):
    def exercise_export(self, target, *, fail_at=None, initial_tf32=True):
        from model_builder.export import exporter

        observations = []
        old_precision = torch.get_float32_matmul_precision()
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.addCleanup(torch.set_float32_matmul_precision, old_precision)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", old_cudnn_tf32)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = initial_tf32

        def observe(stage):
            observations.append(
                (
                    stage,
                    torch.get_float32_matmul_precision(),
                    torch.backends.cudnn.allow_tf32,
                )
            )
            if stage == fail_at:
                raise RuntimeError("injected " + stage + " failure")

        class Model(torch.nn.Module):
            def forward(self, value):
                observe("eager")
                return value + 1

        def export(*args):
            observe("export")
            return object()

        def compile_package(*args, package_path, **kwargs):
            observe("compile")
            Path(package_path).write_bytes(b"stub package")

        def validate(artifact, inputs, device, directory):
            observe("validate")
            return ((inputs[0] + 1).cpu(),)

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(exporter, "_configure_cuda_export_environment"),
            mock.patch.object(exporter, "_strict_export", side_effect=export),
            mock.patch.object(exporter, "_validate_exported_program", return_value=()),
            mock.patch.object(
                torch._inductor, "aoti_compile_and_package", side_effect=compile_package
            ),
            mock.patch.object(
                exporter, "_run_isolated_aoti_package", side_effect=validate
            ),
        ):
            arguments = dict(
                model_name="precision",
                model_version="1",
                input_names=("input",),
                output_names=("output",),
                target=target,
            )
            if fail_at is None:
                exporter.export_package(
                    Model(), (torch.ones(3),), Path(temporary) / "package", **arguments
                )
            else:
                with self.assertRaisesRegex(RuntimeError, "injected " + fail_at):
                    exporter.export_package(
                        Model(),
                        (torch.ones(3),),
                        Path(temporary) / "package",
                        **arguments,
                    )

        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertEqual(torch.backends.cudnn.allow_tf32, initial_tf32)
        return observations

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA required"
    )
    def test_cuda_export_uses_ieee_fp32_and_restores_caller(self):
        for initial_tf32 in (True, False):
            with self.subTest(initial_tf32=initial_tf32):
                observations = self.exercise_export("cuda", initial_tf32=initial_tf32)
                self.assertEqual(
                    observations,
                    [
                        (stage, "highest", False)
                        for stage in ("eager", "export", "compile", "validate")
                    ],
                )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA required"
    )
    def test_cuda_export_restores_caller_after_each_failure(self):
        stages = ("eager", "export", "compile", "validate")
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                observations = self.exercise_export("cuda", fail_at=stage)
                self.assertEqual(
                    observations,
                    [(observed, "highest", False) for observed in stages[: index + 1]],
                )

    def test_cpu_export_preserves_existing_policy(self):
        self.assertEqual(
            self.exercise_export("cpu"),
            [
                (stage, "high", True)
                for stage in ("eager", "export", "compile", "validate")
            ],
        )


if __name__ == "__main__":
    unittest.main()
