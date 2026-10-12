"""A build's requested GPU must match its actual selected CUDA device."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import targets


class CudaDevices:
    """Expose two different devices without requiring a CUDA installation."""

    def __init__(self, available=True):
        self.available = available
        self.queries = []
        self.devices = [((8, 6), "NVIDIA A10G"), ((9, 0), "NVIDIA H100")]

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.devices) if self.available else 0

    def get_device_capability(self, index):
        self.queries.append(("capability", index))
        return self.devices[index][0]

    def get_device_name(self, index):
        self.queries.append(("name", index))
        return self.devices[index][1]


class ProjectTargetTests(unittest.TestCase):
    def test_validation_accepts_static_architectures_and_optional_target(self):
        for device, architecture in (
            ("cpu", None),
            ("cuda", None),
            ("cuda:2", None),
            ("cuda", "sm90"),
            ("cuda:1", "sm86"),
            ("cuda:0", "sm100"),
            ("cuda:0", "sm120"),
        ):
            with self.subTest(device=device, architecture=architecture):
                self.assertIsNone(targets.validate_target(device, architecture))

    def test_invalid_architecture_fails_validation_and_execution(self):
        for architecture in (
            "",
            "90",
            "sm_90",
            "SM90",
            "sm9",
            "sm1000",
            "sm90a",
            "sm９０",
            90,
            [],
        ):
            for function in (targets.validate_target, targets.check_target):
                with self.subTest(architecture=architecture, function=function):
                    with self.assertRaisesRegex(
                        ValueError, "required GPU architecture"
                    ):
                        function("cuda", architecture)

    def test_invalid_device_is_rejected_with_or_without_architecture(self):
        for device in ("", "gpu", "cuda:-1", "cuda:", "cuda:1.0", "cpu:0", None, 0):
            for architecture in (None, "sm90"):
                with self.subTest(device=device, architecture=architecture):
                    with self.assertRaisesRegex(ValueError, "device"):
                        targets.validate_target(device, architecture)

    def test_cpu_cannot_satisfy_required_gpu(self):
        for function in (targets.validate_target, targets.check_target):
            with self.subTest(function=function):
                with self.assertRaisesRegex(ValueError, "CUDA device"):
                    function("cpu", "sm90")

    def test_preflight_and_unconstrained_build_do_not_import_torch(self):
        script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("framework-free target validation imported Torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from model_builder.build.targets import validate_target, check_target
validate_target("cuda:7", "sm90")
assert check_target("cpu", None) is None
assert check_target("cuda:7", None) is None
for device, arch in (("cpu", "sm90"), ("cuda", "sm_90"), ("gpu", None)):
    try:
        check_target(device, arch)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid target was accepted")
assert "torch" not in sys.modules
"""
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                script,
                str(Path(__file__).resolve().parents[1] / "src"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_required_gpu_uses_the_selected_index_and_reports_observed_identity(self):
        for device, architecture, index in (("cuda", "sm86", 0), ("cuda:1", "sm90", 1)):
            cuda = CudaDevices()
            with (
                self.subTest(device=device),
                mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            ):
                self.assertEqual(
                    targets.check_target(device, architecture),
                    {
                        "required_gpu_arch": architecture,
                        "actual_gpu_arch": architecture,
                        "device": device,
                        "gpu_name": cuda.devices[index][1],
                    },
                )
                self.assertEqual(cuda.queries, [("capability", index), ("name", index)])

    def test_wrong_architecture_fails_instead_of_selecting_another_gpu(self):
        cuda = CudaDevices()
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(ValueError, "sm90.*cuda:0.*sm86"):
                targets.check_target("cuda:0", "sm90")
        self.assertIn(("capability", 0), cuda.queries)
        self.assertNotIn(("capability", 1), cuda.queries)

    def test_missing_cuda_fails_without_querying_a_device(self):
        cuda = CudaDevices(available=False)
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(ValueError, "CUDA.*available"):
                targets.check_target("cuda", "sm90")
        self.assertEqual(cuda.queries, [])

    def test_out_of_range_index_fails_before_model_or_device_queries(self):
        cuda = CudaDevices()
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(ValueError, "cuda:2.*2.*visible"):
                targets.check_target("cuda:2", "sm90")
        self.assertEqual(cuda.queries, [])


if __name__ == "__main__":
    unittest.main()
