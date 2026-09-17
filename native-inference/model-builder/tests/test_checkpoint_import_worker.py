"""CPU tensor preservation tests for the standalone PhysicsNeMo importer."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from pnmir_build import checkpoint_import_worker as worker
except ImportError:
    worker = None


@unittest.skipIf(torch is None, "requires the producer Torch environment")
class CheckpointImportWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.checkpoint = self.root / "model.mdlus"
        self.output = self.root / "imported"

        class Model(torch.nn.Module):
            def __init__(self, width=2):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(width))
                self.register_buffer("count", torch.tensor(7, dtype=torch.int64))
                self._args = {"__args__": {"width": width}}

        self.model = Model()
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([1.25, -9.5]))
        self.write_archive(self.model.state_dict())
        self.loader = mock.Mock(return_value=self.model)
        physicsnemo = types.ModuleType("physicsnemo")
        physicsnemo.__version__ = "test-version"
        physicsnemo.Module = types.SimpleNamespace(from_checkpoint=self.loader)
        self.modules = mock.patch.dict(sys.modules, {"physicsnemo": physicsnemo})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def write_archive(self, state, *, legacy=False):
        payload = io.BytesIO()
        torch.save(state, payload)
        if legacy:
            with tarfile.open(self.checkpoint, "w") as archive:
                member = tarfile.TarInfo("model.pt")
                member.size = len(payload.getvalue())
                archive.addfile(member, io.BytesIO(payload.getvalue()))
        else:
            with zipfile.ZipFile(self.checkpoint, "w") as archive:
                archive.writestr("model.pt", payload.getvalue())
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def execute(self, digest=None):
        self.assertTrue(
            callable(getattr(worker, "execute", None)),
            "The importer must expose execute(checkpoint, output, expected_sha256).",
        )
        return worker.execute(self.checkpoint, self.output, digest or self.digest)

    def test_import_preserves_config_and_all_tensor_types_with_strict_reload(self):
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            report = self.execute()
        self.loader.assert_called_once_with(str(self.checkpoint), strict=True)
        self.assertEqual(
            load.call_args.kwargs, {"map_location": "cpu", "weights_only": True}
        )
        state = torch.load(self.output / "checkpoint.pt", weights_only=True)
        self.assertEqual(type(state), dict)
        for name, original in self.model.state_dict().items():
            self.assertTrue(torch.equal(original, state[name]))
            self.assertEqual(original.dtype, state[name].dtype)
            self.assertEqual(state[name].device.type, "cpu")
        self.assertEqual(
            json.loads((self.output / "config.json").read_text()), {"width": 2}
        )
        self.assertEqual(report["status"], "imported")
        self.assertEqual(
            report["verification"],
            {
                "strict_reload": True,
                "tensor_equality": True,
                "source_tensors_unchanged": True,
            },
        )
        self.assertEqual(
            report["checkpoint"],
            {"sha256": self.digest, "size_bytes": self.checkpoint.stat().st_size},
        )
        self.assertEqual(json.loads((self.output / "import.json").read_text()), report)
        for record in report["artifacts"].values():
            payload = (self.output / record["path"]).read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["size_bytes"], len(payload))

    def test_wrong_source_hash_fails_before_checkpoint_loading(self):
        with self.assertRaisesRegex(ValueError, "changed|SHA|sha256"):
            self.execute("0" * 64)
        self.loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_source_mutation_during_loading_cannot_publish_receipt(self):
        def mutate(*args, **kwargs):
            self.checkpoint.write_bytes(b"replacement checkpoint")
            return self.model

        self.loader.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed|SHA|sha256"):
            self.execute()
        self.assertFalse((self.output / "import.json").exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "mine.txt"
        sentinel.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "exists|fresh"):
            self.execute()
        self.assertEqual(sentinel.read_text(), "keep me")
        self.loader.assert_not_called()

    def test_checkpoint_and_output_symlinks_are_rejected(self):
        original = self.checkpoint
        self.checkpoint = self.root / "link.mdlus"
        self.checkpoint.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink|regular"):
            self.execute()
        self.checkpoint = original
        self.output.symlink_to(self.root / "nonexistent", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "exists|symlink|fresh"):
            self.execute()
        self.assertTrue(self.output.is_symlink())

    def test_non_json_constructor_arguments_have_actionable_error(self):
        self.model._args["__args__"]["submodel"] = self.model
        with self.assertRaisesRegex(ValueError, "JSON.*custom|custom.*JSON"):
            self.execute()
        self.assertFalse((self.output / "import.json").exists())

    def test_empty_and_non_tensor_state_cannot_publish_receipt(self):
        for state in ({}, {"weight": object()}):
            with (
                self.subTest(state=state),
                mock.patch.object(self.model, "state_dict", return_value=state),
            ):
                with self.assertRaisesRegex(ValueError, "tensor|empty"):
                    self.execute()
                self.assertFalse(self.output.exists())

    def test_strict_reload_detects_constructor_shape_mismatch(self):
        self.model._args["__args__"]["width"] = 3
        with self.assertRaisesRegex(ValueError, "reconstruct|reload|constructor"):
            self.execute()
        self.assertFalse((self.output / "import.json").exists())

    def test_reconstruction_must_not_silently_cast_checkpoint_dtype(self):
        self.model.double()
        self.write_archive(self.model.state_dict())
        with self.assertRaisesRegex(ValueError, "dtype|tensor"):
            self.execute()
        self.assertFalse((self.output / "import.json").exists())

    def test_initial_loader_must_not_silently_cast_original_archive_tensors(self):
        state = {
            name: tensor.double() if tensor.is_floating_point() else tensor
            for name, tensor in self.model.state_dict().items()
        }
        self.write_archive(state)
        with self.assertRaisesRegex(ValueError, "source|original|cast"):
            self.execute()
        self.assertFalse((self.output / "import.json").exists())

    def test_legacy_tensor_key_renames_preserve_original_tensors(self):
        state = {
            "legacy." + name: tensor for name, tensor in self.model.state_dict().items()
        }
        self.write_archive(state, legacy=True)
        report = self.execute()
        self.assertIs(report["verification"].get("source_tensors_unchanged"), True)

    def test_initial_loader_must_not_change_original_values_or_tensor_count(self):
        for extra in (False, True):
            with self.subTest(extra=extra):
                self.output = self.root / f"imported-{extra}"
                state = dict(self.model.state_dict())
                if extra:
                    state["extra"] = state["weight"].clone()
                else:
                    state["weight"] = state["weight"] + 1
                self.write_archive(state)
                with self.assertRaisesRegex(ValueError, "source|original|cast"):
                    self.execute()
                self.assertFalse(self.output.exists())

    def test_mutating_load_hook_cannot_pass_tensor_equality(self):
        original_load = type(self.model).load_state_dict

        def bad_load(model, *args, **kwargs):
            result = original_load(model, *args, **kwargs)
            with torch.no_grad():
                model.weight.add_(1)
            return result

        with mock.patch.object(type(self.model), "load_state_dict", bad_load):
            with self.assertRaisesRegex(ValueError, "tensor|differ|equal"):
                self.execute()
        self.assertFalse((self.output / "import.json").exists())


if __name__ == "__main__":
    unittest.main()
