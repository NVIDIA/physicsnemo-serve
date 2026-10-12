"""Child-process precision contract; real numerical validation runs on the GPU."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "src/model_builder/export/_isolated_aoti_runner.py"
)


class IsolatedPrecisionTests(unittest.TestCase):
    def run_child(self, target):
        observed = {}
        framework = ModuleType("torch")
        framework.precision = "high"  # NGC's TF32-enabled fresh-process default.
        framework.backends = SimpleNamespace(cudnn=SimpleNamespace(allow_tf32=True))

        class Tensor:
            def to(self, device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

        def select_precision(precision):
            framework.precision = precision

        def load_package(path):
            observed["load_precision"] = framework.precision
            observed["load_cudnn_tf32"] = framework.backends.cudnn.allow_tf32

            def infer(*inputs):
                observed["inference_precision"] = framework.precision
                observed["inference_cudnn_tf32"] = framework.backends.cudnn.allow_tf32
                return Tensor()

            return infer

        device_type, _, index = target.partition(":")
        framework.Tensor = Tensor
        framework.device = lambda value: SimpleNamespace(
            type=device_type, index=int(index) if index else None
        )
        framework.cuda = SimpleNamespace(set_device=mock.Mock())
        framework.set_float32_matmul_precision = select_precision
        framework.load = lambda *args, **kwargs: (Tensor(),)
        framework.save = mock.Mock()
        framework.inference_mode = nullcontext
        inductor = ModuleType("torch._inductor")
        codecache = ModuleType("torch._inductor.codecache")
        inductor.codecache = codecache
        inductor.aoti_load_package = load_package
        framework._inductor = inductor
        modules = {
            "torch": framework,
            "torch._inductor": inductor,
            "torch._inductor.codecache": codecache,
        }
        arguments = [
            str(RUNNER),
            "--package",
            "model.pt2",
            "--inputs",
            "inputs.pt",
            "--outputs",
            "outputs.pt",
            "--target",
            target,
        ]
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(sys, "argv", arguments),
        ):
            spec = importlib.util.spec_from_file_location(
                "isolated_precision_test_runner", RUNNER
            )
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            self.assertEqual(runner.main(), 0)
        framework.save.assert_called_once()
        return observed

    def test_cuda_child_matches_exporter_fp32_policy_before_loading_and_inference(self):
        for target in ("cuda", "cuda:1"):
            with self.subTest(target=target):
                observed = self.run_child(target)
                self.assertEqual(observed["load_precision"], "highest")
                self.assertEqual(observed["inference_precision"], "highest")
                self.assertFalse(observed["load_cudnn_tf32"])
                self.assertFalse(observed["inference_cudnn_tf32"])

    def test_cpu_child_preserves_existing_precision_policy(self):
        observed = self.run_child("cpu")
        self.assertEqual(observed["load_precision"], "high")
        self.assertEqual(observed["inference_precision"], "high")
        self.assertTrue(observed["load_cudnn_tf32"])
        self.assertTrue(observed["inference_cudnn_tf32"])


if __name__ == "__main__":
    unittest.main()
