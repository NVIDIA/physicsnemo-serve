"""The customer example consumes ordinary imported weights without preparation.

The tiny upstream model double and raw/cached tensors below exercise structural
contracts; they do not establish real-checkpoint or scientific CFD parity.
"""

import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import zipfile

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/geotransolver-surface-core"
sys.path.insert(0, str(ROOT / "model-builder/src"))
from model_builder.build import (  # noqa: E402
    authoring,
    authoring_config,
    authoring_sources,
    authoring_worker,
    checkpoint_import_worker,
)


if torch is not None:

    class GlobalTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)

        def forward(self, value):
            return self.projection(value).reshape(1, 1, 1, 2).expand(1, 2, 3, 2)

    class Context(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_tokenizer = GlobalTokenizer()
            self.features = torch.nn.Linear(3, 2)

        def build_context(self, embeddings, positions, geometry, global_embedding):
            local = self.features(positions[0])
            static = (geometry.mean() + positions[0].mean()).expand(1, 2, 3, 4)
            context = torch.cat(
                (static, self.global_tokenizer(global_embedding)), dim=-1
            )
            return context, (local,), None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, streams, context):
            return (streams[0] + self.weight * context.mean(),)

    class GeoTransolverDouble(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self._args = {
                "__name__": "GeoTransolver",
                "__module__": "physicsnemo.experimental.models.geotransolver.geotransolver",
                "__args__": copy.deepcopy(kwargs),
            }
            self.preprocess = torch.nn.ModuleList([torch.nn.Linear(6, 4)])
            self.context_builder = Context()
            self.blocks = torch.nn.ModuleList([Block()])
            self.ln_mlp_out = torch.nn.ModuleList([torch.nn.Linear(6, 4)])

        def forward(self, local_embedding, local_positions, global_embedding, geometry):
            context, features, _ = self.context_builder.build_context(
                (local_embedding,), (local_positions,), geometry, global_embedding
            )
            streams = (
                torch.cat((self.preprocess[0](local_embedding), features[0]), dim=-1),
            )
            for block in self.blocks:
                streams = block(streams, context)
            return self.ln_mlp_out[0](streams[0])


class GeoTransolverFixture:
    def __init__(self, test_case):
        temporary = tempfile.TemporaryDirectory()
        test_case.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model_args = {
            "functional_dim": 6,
            "out_dim": 4,
            "geometry_dim": 3,
            "global_dim": 2,
            "n_layers": 20,
            "n_hidden": 256,
            "dropout": 0.0,
            "n_head": 8,
            "act": "gelu",
            "mlp_ratio": 2,
            "slice_num": 128,
            "use_te": False,
            "plus": False,
            "include_local_features": True,
            "radii": [0.01, 0.05, 0.25, 1.0, 2.5, 5.0],
            "neighbors_in_radius": [4, 8, 16, 64, 128, 256],
            "n_hidden_local": 32,
        }
        torch.manual_seed(17)
        self.official = GeoTransolverDouble(**self.model_args).eval()
        package = ModuleType("physicsnemo")
        package.__version__ = "2.1.1"
        package.Module = mock.Mock()
        package.Module.from_checkpoint.return_value = self.official
        self.loader = package.Module.from_checkpoint
        geotransolver = ModuleType("physicsnemo.experimental.models.geotransolver")
        geotransolver.GeoTransolver = GeoTransolverDouble
        modules = mock.patch.dict(
            sys.modules,
            {
                "physicsnemo": package,
                "physicsnemo.experimental": ModuleType("physicsnemo.experimental"),
                "physicsnemo.experimental.models": ModuleType(
                    "physicsnemo.experimental.models"
                ),
                "physicsnemo.experimental.models.geotransolver": geotransolver,
            },
        )
        modules.start()
        test_case.addCleanup(modules.stop)


def full_inputs(points, geometry_points, case_index, device):
    import torch

    theta = torch.arange(points, dtype=torch.float32, device=device) * (
        2.0 * math.pi / points
    )
    positions = torch.stack(
        (
            0.5 * torch.cos(theta),
            0.25 * torch.sin(theta),
            torch.linspace(-0.5, 0.5, points, device=device),
        ),
        dim=-1,
    ).unsqueeze(0)
    normals = torch.stack(
        (torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1
    ).unsqueeze(0)
    geometry_theta = torch.arange(
        geometry_points, dtype=torch.float32, device=device
    ) * (2.0 * math.pi / geometry_points)
    geometry = torch.stack(
        (
            0.6 * torch.cos(geometry_theta),
            0.3 * torch.sin(geometry_theta),
            torch.linspace(-0.6, 0.6, geometry_points, device=device),
        ),
        dim=-1,
    ).unsqueeze(0)
    if case_index == 2:
        positions = positions * 0.85 + torch.tensor([0.03, -0.02, 0.01], device=device)
        geometry = geometry * 1.15 + torch.tensor([-0.02, 0.01, 0.03], device=device)
    global_embedding = torch.tensor(
        [[[1.1, 35.0] if case_index == 1 else [1.205, 30.0]]],
        dtype=torch.float32,
        device=device,
    )
    return (
        torch.cat((positions, normals), dim=-1),
        positions,
        global_embedding,
        geometry,
    )


def cached_inputs(model, values):
    import torch

    embedding, positions, global_embedding, geometry = values
    context, local_features, _ = model.context_builder.build_context(
        (embedding,), (positions,), geometry, global_embedding
    )
    global_context = model.context_builder.global_tokenizer(global_embedding)
    width = global_context.shape[-1]
    if (
        width <= 0
        or width >= context.shape[-1]
        or not torch.equal(context[..., -width:], global_context)
    ):
        raise ValueError(
            "upstream GeoTransolver context layout does not match the cached-core boundary"
        )
    return (
        embedding,
        local_features[0],
        context[..., :-width].contiguous(),
        global_embedding,
    )


@unittest.skipIf(torch is None, "requires Torch")
class MinimalExampleTests(unittest.TestCase):
    def test_customer_project_selects_exact_backends_and_ten_plugin_assets(self):
        document = json.loads((EXAMPLE / "model-build.json").read_text())
        self.assertEqual(document.get("tensorrt_profile"), "geotransolver-exact-v2")
        self.assertEqual(document["aoti_profile"], "aten-boundary-exact-v2")
        self.assertEqual(document["backends"], ["aoti", "tensorrt"])
        self.assertEqual(document["format_version"], 2)
        names = {
            "linear",
            "gemm",
            "token_sum",
            "slice_bmm",
            "layer_norm",
            "softmax",
            "attention",
            "gelu",
            "weighted_blend",
            "deslice_bmm",
        }
        self.assertEqual(
            set(document["assets"]), {f"tensorrt_exact_{name}_plugin" for name in names}
        )

    def setUp(self):
        self.fixture = GeoTransolverFixture(self)
        spec = importlib.util.spec_from_file_location(
            "minimal_geo", EXAMPLE / "adapter.py"
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.config = self.fixture.model_args

    def model(self):
        try:
            return self.adapter.create_model(self.config, {})
        except ValueError as exc:
            self.fail(
                f"Plain imported constructor config must work without fixtures: {exc}"
            )

    def test_plain_imported_constructor_and_weights_preserve_full_model_core(self):
        core = self.model().eval()
        full = self.fixture.official
        core.load_state_dict(full.state_dict(), strict=True)
        self.assertIsInstance(core, GeoTransolverDouble)
        self.assertEqual(set(core.state_dict()), set(full.state_dict()))
        self.assertFalse(core._load_state_dict_post_hooks)
        with torch.no_grad():
            for index in range(3):
                raw = full_inputs(8, 16, index, "cpu")
                cached = cached_inputs(full, raw)
                self.assertTrue(torch.equal(core(*cached), full(*raw)))

    def test_cases_are_repeatable_and_have_the_imported_surface_dimensions(self):
        state = torch.get_rng_state().clone()
        try:
            cases = self.adapter.create_cases(self.config, {})
        except ValueError as exc:
            self.fail(f"Synthetic core cases must not require prepared assets: {exc}")
        self.assertEqual(len(cases), 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        again = self.adapter.create_cases(self.config, {})
        for case, repeated in zip(cases, again, strict=True):
            self.assertEqual(
                [list(x.shape) for x in case],
                [[1, 32, 6], [1, 32, 192], [1, 8, 128, 224], [1, 1, 2]],
            )
            for value, other in zip(case, repeated, strict=True):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.equal(value, other))
        for column in range(4):
            self.assertFalse(torch.equal(cases[0][column], cases[1][column]))

    def test_generic_import_preserves_constructor_state_and_checkpoint_identity(self):
        checkpoint = self.fixture.root / "model.mdlus"
        payload = io.BytesIO()
        torch.save(self.fixture.official.state_dict(), payload)
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("model.pt", payload.getvalue())
            archive.writestr("args.json", json.dumps(self.fixture.official._args))
            archive.writestr("metadata.json", '{"physicsnemo_version":"2.2.0a0"}')
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = self.fixture.root / "imported"
        result = checkpoint_import_worker.execute(checkpoint, output, digest)
        self.fixture.loader.assert_called_once_with(str(checkpoint), strict=True)
        self.assertEqual(json.loads((output / "config.json").read_text()), self.config)
        state = torch.load(
            output / "checkpoint.pt", weights_only=True, map_location="cpu"
        )
        self.assertEqual(set(state), set(self.fixture.official.state_dict()))
        for name, expected in self.fixture.official.state_dict().items():
            self.assertEqual(state[name].dtype, expected.dtype)
            self.assertEqual(state[name].shape, expected.shape)
            self.assertTrue(torch.equal(state[name], expected))
        self.assertEqual(result["status"], "imported")
        self.assertEqual(result["checkpoint"]["sha256"], digest)
        self.assertEqual(result["environment"]["physicsnemo"], "2.1.1")
        self.assertEqual(
            result["verification"],
            {
                "strict_reload": True,
                "tensor_equality": True,
                "source_tensors_unchanged": True,
            },
        )
        for record in result["artifacts"].values():
            data = (output / record["path"]).read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(record["size_bytes"], len(data))

    def snapshot_project(self):
        # Same upstream API double with dimensions matching its tiny layers.
        self.config.update(
            n_hidden=4,
            n_head=2,
            slice_num=3,
            n_hidden_local=2,
            radii=[0.1],
            neighbors_in_radius=[4],
        )
        project_dir = self.fixture.root / "project"
        project_dir.mkdir()
        for name in ("adapter.py", "model-build.json"):
            (project_dir / name).write_bytes((EXAMPLE / name).read_bytes())
        weights = project_dir / "weights"
        weights.mkdir()
        (weights / "config.json").write_text(json.dumps(self.config))
        torch.save(self.fixture.official.state_dict(), weights / "checkpoint.pt")
        document = json.loads((project_dir / "model-build.json").read_text())
        self.assertEqual(document["config"], "weights/config.json")
        self.assertEqual(document["checkpoint"], "weights/checkpoint.pt")
        self.assertNotIn("fixtures", document["assets"])
        # Check captures SDK library identities but does not load native plugins.
        for name, relative in document["assets"].items():
            path = project_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"SDK plugin identity fixture: {name}".encode())
        project = authoring_config.load(project_dir)
        source, selected, configuration = authoring._selected(project)
        snapshot = self.fixture.root / "snapshot"
        authoring._snapshot(
            snapshot,
            project,
            source,
            selected,
            configuration,
            {"adapter": "adapter.py"},
        )
        return snapshot

    def check_snapshot(self, snapshot, output="checked"):
        # Production uses one worker process per project; isolate its imports here.
        with (
            mock.patch.object(authoring_sources, "_TREES", {}),
            mock.patch.object(authoring_sources, "_IMPORT_ROOTS", set()),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.dict(sys.modules),
        ):
            try:
                return authoring_worker.execute(
                    snapshot, self.fixture.root / output, "check", "cpu"
                )
            finally:
                for temporary, _, _ in authoring_sources._TREES.values():
                    temporary.cleanup()

    def test_generic_authoring_check_uses_only_imported_config_and_checkpoint(self):
        result = self.check_snapshot(self.snapshot_project())
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(
            [entry["name"] for entry in result["tensor_contract"]["inputs"]],
            ["local_embedding", "local_features", "static_context", "global_embedding"],
        )
        self.assertEqual(
            result["tensor_contract"]["outputs"],
            [
                {
                    "name": "surface_fields_standardized",
                    "dtype": "float32",
                    "shape": [1, 32, 4],
                }
            ],
        )

    def test_original_project_changes_do_not_change_captured_check(self):
        snapshot = self.snapshot_project()
        project = self.fixture.root / "project"
        (project / "adapter.py").write_text(
            "raise AssertionError('uncaptured adapter')\n"
        )
        (project / "weights/config.json").write_text("{}\n")
        (project / "weights/checkpoint.pt").write_bytes(b"replacement weights")
        result = self.check_snapshot(snapshot)
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["tensor_contract"]["outputs"][0]["shape"], [1, 32, 4])

    def test_changed_captured_inputs_fail_before_model_execution(self):
        snapshot = self.snapshot_project()
        manifest = json.loads((snapshot / "snapshot.json").read_text())
        for index, record in enumerate(manifest["files"]):
            path = snapshot / record["path"]
            original = path.read_bytes()
            with self.subTest(path=record["path"]):
                path.write_bytes(original + b"changed")
                try:
                    with mock.patch.object(
                        authoring_worker.worker, "_prepare_model"
                    ) as prepare:
                        with self.assertRaisesRegex(
                            ValueError, "Captured input changed"
                        ):
                            self.check_snapshot(snapshot, f"failed-{index}")
                        prepare.assert_not_called()
                    report = json.loads(
                        (self.fixture.root / f"failed-{index}/check.json").read_text()
                    )
                    self.assertEqual(report["status"], "failed")
                finally:
                    path.write_bytes(original)

    def test_incompatible_surface_configuration_is_rejected(self):
        self.model()
        for key, value in (
            ("functional_dim", [6, 6]),
            ("include_local_features", False),
            ("use_te", True),
            ("guard_config", {"enabled": True}),
            ("structured_shape", [4, 8]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "surface"):
                self.adapter.create_model(dict(self.config, **{key: value}), {})


if __name__ == "__main__":
    unittest.main()
