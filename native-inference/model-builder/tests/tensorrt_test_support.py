"""TensorRT plugin inputs, exporter doubles, and profile publication fixtures.

Torch remains lazy: only TensorRTPayloadFixture setup imports its builder.
"""

from contextlib import nullcontext
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

from model_builder.export import tensorrt_profiles
from model_builder.export import tensorrt_profiles as profiles
import authoring_test_support
import worker_test_support

EXACT = "layout-order-exact"

CREATORS = (
    "PNMIRExactLinear",
    "PNMIRExactGemm",
    "PNMIRExactTokenSum",
    "PNMIRExactSliceBmm",
    "PNMIRExactLayerNorm",
    "PNMIRExactSoftmax",
    "PNMIRExactAttention",
    "PNMIRExactGelu",
)

TRANSFORMS = (
    ("exact_linear", "_replace_linear_subgraphs"),
    ("exact_gemm", "_replace_constant_rhs_matmuls"),
    ("exact_layer_norm", "_replace_layer_norms"),
    ("exact_attention", "_replace_attention_subgraphs"),
    ("exact_token_sum", "_replace_token_sums"),
    ("exact_slice_bmm", "_replace_slice_bmms"),
    ("exact_gelu", "_replace_gelu_subgraphs"),
    ("exact_softmax", "_replace_softmaxes"),
)

PLUGINS = (
    "exact_linear",
    "exact_gemm",
    "exact_token_sum",
    "exact_slice_bmm",
    "exact_layer_norm",
    "exact_softmax",
    "exact_attention",
    "exact_gelu",
)


class TensorRTPluginFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.libraries = {}
        for name in tensorrt_profiles.EXACT_PLUGIN_NAMES:
            path = self.root / (name + ".so")
            path.write_bytes(("library identity fixture: " + name).encode())
            self.libraries[name] = path


class TensorRTPayloadFixture:
    def setUp(self):
        from model_builder.export import tensorrt_builder

        self.module = tensorrt_builder
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.onnx"
        self.source.write_bytes(b"test graph; not executable ONNX")
        self.output = self.root / "tensorrt"
        self.onnx = SimpleNamespace(
            __version__="test-onnx",
            load_model=mock.Mock(return_value=object()),
            checker=SimpleNamespace(check_model=mock.Mock()),
        )
        self.config = SimpleNamespace(
            set_memory_pool_limit=mock.Mock(), clear_flag=mock.Mock()
        )
        self.builder = SimpleNamespace(
            create_network=mock.Mock(return_value=object()),
            create_builder_config=mock.Mock(return_value=self.config),
            build_serialized_network=mock.Mock(return_value=b"test compiled plan"),
        )
        self.trt = SimpleNamespace(
            __version__="test-tensorrt",
            Logger=mock.Mock(WARNING=2),
            Builder=mock.Mock(return_value=self.builder),
            OnnxParser=mock.Mock(
                return_value=SimpleNamespace(
                    parse_from_file=mock.Mock(return_value=True)
                )
            ),
            init_libnvinfer_plugins=mock.Mock(return_value=True),
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
            MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
            BuilderFlag=SimpleNamespace(TF32="tf32"),
        )
        self.contract = (
            [{"name": "input", "dtype": "float32", "shape": [4]}],
            [{"name": "output", "dtype": "float32", "shape": [4]}],
        )

    def build(self, **kwargs):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            set_device=mock.Mock(),
            get_device_capability=lambda index: (9, 0),
            get_device_name=lambda index: "test GPU",
        )
        with (
            mock.patch.object(self.module.torch, "cuda", cuda),
            mock.patch.object(self.module, "_onnx_module", return_value=self.onnx),
            mock.patch.object(
                self.module, "_graph_contract", return_value=self.contract
            ),
            mock.patch.object(self.module, "_tensorrt_module", return_value=self.trt),
        ):
            result = self.module.build_tensorrt_package(
                self.source,
                self.output,
                model_name="affine",
                model_version="test",
                device=1,
                workspace_size=4096,
                **kwargs,
            )
        cuda.set_device.assert_called_once_with(1)
        return result


class TensorRTRecipeFixture:
    def fixture(self, fixture_type):
        fixture = fixture_type()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        return fixture

    def backend_modules(self):
        modules = worker_test_support.backend_modules()
        modules["torch"].cuda = SimpleNamespace(
            current_device=lambda: 0, set_device=mock.Mock()
        )
        modules["torch"].device = lambda device: device
        modules[
            "model_builder.export.onnx_exporter"
        ].export_onnx_model.return_value = Path("model.onnx")
        modules[
            "model_builder.export.tensorrt_exporter"
        ].tensorrt_ieee_fp32 = nullcontext
        return modules

    def exact_project(self):
        # Asset and lock checks use the mocked Linux container on every host.
        for patcher in (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch("os.getuid", return_value=1000, create=True),
            mock.patch("os.getgid", return_value=1000, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        fixture = self.fixture(authoring_test_support.AuthoringContainerFixture)
        fixture.document["tensorrt_profile"] = EXACT
        for name in PLUGINS:
            asset = "tensorrt_" + name + "_plugin"
            path = fixture.project / (name + ".so")
            path.write_bytes((name + " test plugin input").encode())
            fixture.document["assets"][asset] = path.name
        fixture.write_project()
        return fixture


class ProfileBuildFixture:
    def fixture(self, kind=worker_test_support.WorkerFixture):
        fixture = kind()
        fixture.addCleanup = self.addCleanup
        if hasattr(fixture, "setUp"):
            fixture.setUp()
        return fixture

    def build(self, profile, *, difference=False, signed_zero=False):
        fixture = self.fixture()
        fixture.recipe["tensorrt_profile"] = profile
        fixture.recipe_path.write_text(json.dumps(fixture.recipe))
        if difference:
            values = list(
                struct.unpack("<4f", fixture.prepared["references"][0][0]["data"])
            )
            values[0] += 1.0e-6
            fixture.prepared["references"][0][0]["data"] = struct.pack("<4f", *values)
        if signed_zero:
            fixture.prepared["inputs"][0] = (
                worker_test_support.tensor("input", [-0.5, 0.0, 1.0, 2.0]),
            )
            fixture.prepared["references"][0] = (
                worker_test_support.tensor("output", [-0.0, 1.0, 3.0, 5.0]),
            )
        # Separate the native gate behavior from profile schema acceptance.
        with mock.patch.object(
            profiles, "validate_tensorrt_profile", side_effect=lambda name: name
        ):
            report = fixture.run_build(["tensorrt"])
        return fixture, report
