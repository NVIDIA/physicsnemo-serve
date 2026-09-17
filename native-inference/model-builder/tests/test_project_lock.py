"""Project identity locks cannot silently refresh inputs or lose profiles."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import project_lock


class ProjectLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "model-build.lock.json"
        self.identity = {
            "checkpoint_sha256": "a" * 64,
            "backends": ["aoti"],
            "device": "cuda",
        }

    def inspect(self, key="build:default", identity=None, **kwargs):
        return project_lock.inspect_lock(
            self.path, key, self.identity if identity is None else identity, **kwargs
        )

    def write(self, entries=None):
        content = {
            "format_version": 1,
            "entries": entries
            if entries is not None
            else {"build:default": self.identity},
        }
        self.path.write_text(json.dumps(content))
        return self.path.read_bytes()

    def test_inspection_is_read_only_and_copies_selected_identity(self):
        inspected = self.inspect()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(
            inspected.get("content"),
            {"format_version": 1, "entries": {"build:default": self.identity}},
        )
        self.assertIsNone(inspected.get("original_sha256"))
        self.identity["backends"].append("tensorrt")
        self.assertEqual(
            inspected["content"]["entries"]["build:default"]["backends"], ["aoti"]
        )

    def test_publish_returns_identity_and_unchanged_lock_keeps_original_bytes(self):
        original = self.write()
        inspected = self.inspect()
        record = project_lock.publish_lock(self.path, inspected)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(
            {key: record[key] for key in ("path", "sha256", "size_bytes")},
            {
                "path": str(self.path),
                "sha256": hashlib.sha256(original).hexdigest(),
                "size_bytes": len(original),
            },
        )
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_publication_returns_exact_snapshot_text_for_provenance(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.write()
                record = project_lock.publish_lock(self.path, self.inspect())
                original = self.path.read_bytes()
                self.path.write_text("changed after publication")
                self.assertIsInstance(record.get("serialized_text"), str)
                self.assertEqual(record["serialized_text"].encode("utf-8"), original)
                self.assertEqual(hashlib.sha256(original).hexdigest(), record["sha256"])
                self.assertEqual(len(original), record["size_bytes"])

    def test_new_profile_preserves_existing_profile(self):
        self.write()
        second = {"checkpoint_sha256": "b" * 64, "backends": ["aoti", "tensorrt"]}
        inspected = self.inspect("build:h100", second)
        project_lock.publish_lock(self.path, inspected)
        self.assertEqual(
            json.loads(self.path.read_bytes())["entries"],
            {"build:default": self.identity, "build:h100": second},
        )

    def test_changed_identity_requires_explicit_update(self):
        original = self.write()
        changed = dict(self.identity, checkpoint_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "changed|differ|update"):
            self.inspect(identity=changed)
        self.assertEqual(self.path.read_bytes(), original)

    def test_refresh_updates_only_selected_profile(self):
        other = {"checkpoint_sha256": "c" * 64}
        self.write({"build:default": self.identity, "build:h100": other})
        changed = dict(self.identity, device="cpu")
        inspected = self.inspect(identity=changed, update=True)
        project_lock.publish_lock(self.path, inspected)
        self.assertEqual(
            json.loads(self.path.read_bytes())["entries"],
            {"build:default": changed, "build:h100": other},
        )

    def test_boolean_and_integer_identity_are_different(self):
        self.write({"build:default": {"selection": True}})
        with self.assertRaisesRegex(ValueError, "changed|differ|update"):
            self.inspect(identity={"selection": 1})

    def test_invalid_locks_are_preserved_even_with_update(self):
        invalid = [
            '{"format_version":1,"entries":{},"extra":true}',
            '{"format_version":true,"entries":{}}',
            '{"format_version":2,"entries":{}}',
            '{"format_version":1,"entries":{},"entries":{}}',
            '{"format_version":1,"entries":{"build:a":{"x":1,"x":2}}}',
            '{"format_version":1,"entries":{"build:a":{"x":NaN}}}',
            '{"format_version":1,"entries":{"build:a":{"x":1e999}}}',
            '{"format_version":1,"entries":{"build:a":[]}}',
            '{"format_version":1,"entries":[]}',
            '{"format_version":1}',
            "[]",
            "",
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.path.write_text(raw)
                with self.assertRaises(ValueError):
                    self.inspect(update=True)
                self.assertEqual(self.path.read_text(), raw)

    def test_invalid_request_is_rejected_without_writing(self):
        for identity in (
            [],
            {"x": float("nan")},
            {"x": float("inf")},
            {1: "x"},
            {"x": (1, 2)},
        ):
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self.inspect(identity=identity)
        for key in ("", " ", 3):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.inspect(key=key)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_publish_rejects_changed_or_newly_created_lock(self):
        for existed in (False, True):
            with self.subTest(existed=existed):
                if self.path.exists():
                    self.path.unlink()
                if existed:
                    self.write()
                inspected = self.inspect("build:h100")
                current = self.write({"build:external": {"source": "editor"}})
                with self.assertRaisesRegex(ValueError, "changed|concurrent"):
                    project_lock.publish_lock(self.path, inspected)
                self.assertEqual(self.path.read_bytes(), current)

    def test_stale_publish_cannot_lose_an_added_profile(self):
        first = self.inspect("build:first")
        second = self.inspect("build:second")
        project_lock.publish_lock(self.path, first)
        with self.assertRaisesRegex(ValueError, "changed|concurrent"):
            project_lock.publish_lock(self.path, second)
        self.assertEqual(
            set(json.loads(self.path.read_bytes())["entries"]), {"build:first"}
        )

    def test_publish_rejects_mutated_inspection_and_different_destination(self):
        inspected = self.inspect()
        mutated = copy.deepcopy(inspected)
        mutated["content"]["entries"]["build:default"]["device"] = "cpu"
        with self.assertRaises(ValueError):
            project_lock.publish_lock(self.path, mutated)
        with self.assertRaises(ValueError):
            project_lock.publish_lock(self.root / "other.json", inspected)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_symlink_directory_and_dangling_symlink_are_preserved(self):
        original = self.root / "original.json"
        original.write_text('{"format_version":1,"entries":{}}')
        for kind in ("file", "directory", "dangling"):
            with self.subTest(kind=kind):
                inspected = self.inspect()
                if kind == "directory":
                    self.path.mkdir()
                else:
                    self.path.symlink_to(
                        original if kind == "file" else self.root / "missing"
                    )
                with self.assertRaises(ValueError):
                    self.inspect(update=True)
                with self.assertRaises(ValueError):
                    project_lock.publish_lock(self.path, inspected)
                if kind == "directory":
                    self.path.rmdir()
                else:
                    self.path.unlink()
                self.assertEqual(
                    original.read_text(), '{"format_version":1,"entries":{}}'
                )

    def test_simultaneous_publish_has_one_winner_without_lost_entries(self):
        first = self.inspect("build:first")
        second = self.inspect("build:second")
        ready = threading.Barrier(2)

        def publish(inspected):
            ready.wait(timeout=5)
            try:
                project_lock.publish_lock(self.path, inspected)
                return "published"
            except ValueError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(publish, value) for value in (first, second)]
            self.assertEqual(
                sorted(future.result(timeout=10) for future in futures),
                ["published", "rejected"],
            )
        self.assertEqual(len(json.loads(self.path.read_bytes())["entries"]), 1)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_atomic_replace_failure_keeps_original_and_releases_writer_guard(self):
        original = self.write()
        inspected = self.inspect("build:h100")
        with mock.patch("os.replace", side_effect=OSError("disk write failed")):
            with self.assertRaisesRegex(OSError, "disk write failed"):
                project_lock.publish_lock(self.path, inspected)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])
        project_lock.publish_lock(self.path, inspected)
        self.assertIn("build:h100", json.loads(self.path.read_bytes())["entries"])


if __name__ == "__main__":
    unittest.main()
