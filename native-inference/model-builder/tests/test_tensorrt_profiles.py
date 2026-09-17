"""Framework-free profile validation and explicit native plugin registration."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_export import tensorrt_exact_graphs, tensorrt_profiles


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


class TensorRTProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.libraries = {}
        for name in tensorrt_profiles.EXACT_PLUGIN_NAMES:
            path = self.root / (name + ".so")
            path.write_bytes(("library identity fixture: " + name).encode())
            self.libraries[name] = path

    def test_preflight_and_operator_contract_need_no_optional_framework(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from pnmir_export import tensorrt_profiles as p; "
                "assert p.validate_tensorrt_profile('baseline') == 'baseline'; "
                "assert p.validate_tensorrt_profile('layout-order-exact') == 'layout-order-exact'; "
                "assert len(p.required_operators()) == 8; "
                "assert not {'torch', 'onnx', 'numpy', 'tensorrt'} & sys.modules.keys()",
                str(Path(__file__).resolve().parents[1] / "src"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_profile_is_rejected_without_loading_native_libraries(self):
        with mock.patch.object(tensorrt_profiles.ctypes, "CDLL") as load:
            for value in (None, "", "unknown", [], {}, 1):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "tensorrt_profile"):
                        tensorrt_profiles.resolve_plugin_libraries(
                            value, self.libraries
                        )
            load.assert_not_called()

    def test_baseline_has_no_plugins_and_rejects_unused_libraries(self):
        self.assertEqual(tensorrt_profiles.resolve_plugin_libraries("baseline"), {})
        with self.assertRaisesRegex(ValueError, "baseline.*plugin"):
            tensorrt_profiles.resolve_plugin_libraries("baseline", self.libraries)

    def test_exact_requires_complete_named_set_and_existing_files(self):
        incomplete = dict(self.libraries)
        incomplete.pop("exact_attention")
        for libraries in (None, {}, incomplete, {**self.libraries, "typo": "other.so"}):
            with self.subTest(libraries=libraries):
                with self.assertRaisesRegex(ValueError, "all eight plugin"):
                    tensorrt_profiles.resolve_plugin_libraries(EXACT, libraries)
        self.assertEqual(
            tensorrt_profiles.resolve_plugin_libraries(EXACT, self.libraries),
            self.libraries,
        )
        self.libraries["exact_attention"].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "exact_attention"):
            tensorrt_profiles.resolve_plugin_libraries(EXACT, self.libraries)

    def test_registry_metadata_binds_all_library_bytes_and_keeps_handles(self):
        library_handles = [object() for _ in CREATORS]
        registry = SimpleNamespace(get_creator=mock.Mock(return_value=object()))
        trt = SimpleNamespace(get_plugin_registry=lambda: registry)
        with mock.patch.object(
            tensorrt_profiles.ctypes, "CDLL", side_effect=library_handles
        ) as load:
            handles, records = tensorrt_profiles.load_exact_plugins(trt, self.libraries)
        self.assertEqual(handles, library_handles)
        self.assertEqual(
            load.call_args_list,
            [
                mock.call(str(path), mode=tensorrt_profiles.ctypes.RTLD_GLOBAL)
                for path in self.libraries.values()
            ],
        )
        self.assertEqual(
            registry.get_creator.call_args_list,
            [mock.call(creator, "1", "") for creator in CREATORS],
        )
        self.assertEqual(
            records,
            {
                name: {
                    "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in self.libraries.items()
            },
        )

    def test_missing_creator_fails_without_loading_remaining_plugins(self):
        registry = SimpleNamespace(get_creator=mock.Mock(side_effect=[object(), None]))
        trt = SimpleNamespace(get_plugin_registry=lambda: registry)
        with mock.patch.object(tensorrt_profiles.ctypes, "CDLL") as load:
            with self.assertRaisesRegex(RuntimeError, "PNMIRExactGemm"):
                tensorrt_profiles.load_exact_plugins(trt, self.libraries)
        self.assertEqual(load.call_count, 2)

    def test_required_operators_match_sdk_abi_and_return_fresh_descriptors(self):
        operators = tensorrt_profiles.required_operators()
        self.assertEqual(
            operators,
            [
                {"id": "pnmir.tensorrt-" + name.replace("_", "-"), "abi": "1"}
                for name in self.libraries
            ],
        )
        operators[0]["abi"] = "changed"
        self.assertEqual(tensorrt_profiles.required_operators()[0]["abi"], "1")

    def test_transform_order_preserves_attention_and_slice_layout_patterns(self):
        onnx, model = object(), object()
        calls = []
        mocks = {}
        for name, symbol in TRANSFORMS:

            def transform(actual_onnx, actual_model, *, name=name):
                self.assertIs(actual_onnx, onnx)
                self.assertIs(actual_model, model)
                calls.append(name)
                return len(calls)

            mocks[symbol] = transform
        with mock.patch.multiple(tensorrt_exact_graphs, **mocks):
            counts = tensorrt_profiles.prepare_exact_graph(onnx, model)
        self.assertEqual(calls, [name for name, _ in TRANSFORMS])
        self.assertEqual(
            counts, {name: index + 1 for index, (name, _) in enumerate(TRANSFORMS)}
        )

    def test_unsupported_graph_cannot_claim_exact_profile(self):
        mocks = {symbol: mock.Mock(return_value=1) for _, symbol in TRANSFORMS}
        mocks["_replace_slice_bmms"].return_value = 0
        with mock.patch.multiple(tensorrt_exact_graphs, **mocks):
            with self.assertRaisesRegex(ValueError, "no supported exact_slice_bmm"):
                tensorrt_profiles.prepare_exact_graph(object(), object())
        mocks["_replace_gelu_subgraphs"].assert_not_called()
        mocks["_replace_softmaxes"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
