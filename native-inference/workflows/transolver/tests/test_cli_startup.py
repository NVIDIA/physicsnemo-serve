"""Exercise the production CLI startup policy without a native engine or CUDA."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


class CliStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != "posix":
            raise unittest.SkipTest("the production CLI requires POSIX setenv")
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required for CLI checks")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        build_root = Path(temporary.name)
        stub = build_root / "runner.cpp"
        # Replace only the GPU-backed Runner boundary. Compile the production
        # main.cpp unchanged so argument parsing, startup ordering, and report
        # publication all execute through the real CLI control flow.
        stub.write_text(
            r"""
#include "transolver_workflow/runner.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace transolver_workflow {
namespace {
void record(const std::string& event) {
  std::ofstream stream(std::getenv("PNMS_CLI_TEST_EVENTS"), std::ios::app);
  stream << event << '\n';
}
}

class Runner::Impl {
 public:
  explicit Impl(RunnerConfig value) : config(std::move(value)) {}
  RunnerConfig config;
};

Runner::Runner(RunnerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) { record("construct"); }
Runner::~Runner() = default;

std::string Runner::self_test(const CancellationToken&) {
  const auto& fixture = impl_->config.startup_fixture_path;
  record("self_test:" + fixture);
  if (!fixture.empty()) {
    std::ifstream stream(fixture);
    std::string value;
    stream >> value;
    if (!stream || value != "match") {
      throw std::runtime_error("startup fixture mismatch");
    }
  }
  return R"({"status":"passed"})";
}

RunnerResult Runner::run(const RunnerRequest& request, const CancellationToken&) {
  record("run");
  if (!request.dump_input_dir.empty()) {
    std::filesystem::create_directories(request.dump_input_dir);
    for (const auto* name : {"fx.f32", "embedding.f32", "permutation.i64", "source_cell_ids.i64"})
      std::ofstream(std::filesystem::path(request.dump_input_dir) / name) << "diagnostic dump\n";
  }
  if (!request.standardized_output.empty())
    std::ofstream(request.standardized_output) << "published fields\n";
  if (!request.physical_output.empty())
    std::ofstream(request.physical_output) << "published physical fields\n";
  std::ofstream(request.metadata_path) << "published metadata\n";
  return {};
}
}
"""
        )
        cls.executable = build_root / "physicsnemo-transolver"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-I",
                str(WORKFLOW_ROOT / "include"),
                str(WORKFLOW_ROOT / "src" / "main.cpp"),
                str(stub),
                "-o",
                str(cls.executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / "events.txt"
        self.output = self.root / "fields.f32"
        self.metadata = self.root / "fields.f32.json"
        self.report = self.root / "startup.json"
        self.fixture = self.root / "fixture.json"
        self.fixture.write_text("match\n")

    def run_cli(self, *options, startup_only=False):
        command = [
            str(self.executable),
            "--package",
            "model.pt2",
            "--domain",
            "surface",
        ]
        if startup_only:
            command.extend(["--startup-only", "true"])
        else:
            command.extend(
                [
                    "--mesh",
                    "mesh.vtp",
                    "--stl",
                    "body.stl",
                    "--standardized-output",
                    str(self.output),
                ]
            )
        return subprocess.run(
            [*command, *map(str, options)],
            cwd=self.root,
            env={**os.environ, "PNMS_CLI_TEST_EVENTS": str(self.events)},
            capture_output=True,
            text=True,
        )

    def recorded_events(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def assert_report_rejected(self, result, message):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertIn("--startup-report", result.stderr)
        self.assertEqual(self.recorded_events(), [])

    def test_bad_fixture_without_report_fails_before_running(self):
        self.fixture.write_text("mismatch\n")
        result = self.run_cli("--startup-fixture", self.fixture)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("startup fixture mismatch", result.stderr)
        self.assertEqual(
            self.recorded_events(), ["construct", f"self_test:{self.fixture}"]
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())
        self.assertFalse(self.report.exists())

    def test_valid_fixture_without_report_checks_before_running(self):
        result = self.run_cli("--startup-fixture", self.fixture)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.recorded_events(), ["construct", f"self_test:{self.fixture}", "run"]
        )
        self.assertEqual(self.output.read_text(), "published fields\n")
        self.assertTrue(self.metadata.is_file())
        self.assertFalse(self.report.exists())

    def test_legacy_run_without_fixture_or_report(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["construct", "run"])
        self.assertTrue(self.output.is_file())

    def test_report_option_checks_and_writes_report_before_running(self):
        result = self.run_cli("--startup-report", self.report)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["construct", "self_test:", "run"])
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})
        self.assertTrue(self.output.is_file())

    def test_startup_only_with_report_checks_without_running(self):
        result = self.run_cli(
            "--startup-fixture",
            self.fixture,
            "--startup-report",
            self.report,
            startup_only=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.recorded_events(), ["construct", f"self_test:{self.fixture}"]
        )
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_startup_only_still_requires_report_with_fixture(self):
        result = self.run_cli("--startup-fixture", self.fixture, startup_only=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("--startup-only requires --startup-report", result.stderr)
        self.assertEqual(self.recorded_events(), [])

    def test_report_cannot_share_standardized_output(self):
        result = self.run_cli("--startup-report", self.output)
        self.assert_report_rejected(result, "output paths must be distinct")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_report_cannot_share_physical_output(self):
        physical = self.root / "physical.f32"
        result = self.run_cli(
            "--physical-output",
            physical,
            "--stats",
            self.fixture,
            "--startup-report",
            physical,
        )
        self.assert_report_rejected(result, "output paths must be distinct")
        self.assertFalse(physical.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_report_cannot_share_explicit_or_default_metadata(self):
        for options in ((), ("--metadata", self.metadata)):
            with self.subTest(options=options):
                result = self.run_cli(*options, "--startup-report", self.metadata)
                self.assert_report_rejected(result, "output paths must be distinct")
                self.assertFalse(self.output.exists())
                self.assertFalse(self.metadata.exists())

    def test_report_rejects_normalized_output_alias(self):
        (self.root / "nested").mkdir()
        report = self.root / "nested" / ".." / self.output.name
        result = self.run_cli("--startup-report", report)
        self.assert_report_rejected(result, "output paths must be distinct")
        self.assertFalse(self.output.exists())

    def test_report_rejects_symlinked_parent_output_alias(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        result = self.run_cli("--startup-report", alias / self.output.name)
        self.assert_report_rejected(result, "output paths must be distinct")
        self.assertFalse(self.output.exists())

    def test_report_preserves_existing_file(self):
        original = b"existing report must survive\n"
        self.report.write_bytes(original)
        result = self.run_cli("--startup-report", self.report)
        self.assertEqual(self.report.read_bytes(), original)
        self.assert_report_rejected(result, "already exists")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.metadata.exists())

    def test_report_preserves_authored_fixture(self):
        original = self.fixture.read_bytes()
        result = self.run_cli(
            "--startup-fixture",
            self.fixture,
            "--startup-report",
            self.fixture,
            startup_only=True,
        )
        self.assertEqual(self.fixture.read_bytes(), original)
        self.assert_report_rejected(result, "already exists")

    def test_report_preserves_existing_fixture_symlink_and_hardlink(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                self.fixture.write_text("match\n")
                original = self.fixture.read_bytes()
                report = self.root / f"{kind}.json"
                if kind == "symlink":
                    report.symlink_to(self.fixture)
                else:
                    report.hardlink_to(self.fixture)
                result = self.run_cli(
                    "--startup-fixture",
                    self.fixture,
                    "--startup-report",
                    report,
                    startup_only=True,
                )
                self.assertEqual(self.fixture.read_bytes(), original)
                self.assertEqual(report.read_bytes(), original)
                self.assert_report_rejected(result, "already exists")

    def test_report_rejects_existing_directory_before_constructing_runner(self):
        self.report.mkdir()
        result = self.run_cli("--startup-report", self.report)
        self.assert_report_rejected(result, "already exists")
        self.assertEqual(list(self.report.iterdir()), [])

    def test_report_rejects_dangling_final_and_parent_symlinks(self):
        for parent in (False, True):
            with self.subTest(parent=parent):
                target = self.root / f"missing-{parent}"
                alias = self.root / f"alias-{parent}"
                alias.symlink_to(target, target_is_directory=parent)
                report = alias / "startup.json" if parent else alias
                result = self.run_cli("--startup-report", report)
                self.assert_report_rejected(result, "contains a dangling symlink")
                self.assertFalse(target.exists())
                self.assertTrue(alias.is_symlink())

    def test_startup_only_report_can_match_inactive_outputs(self):
        result = self.run_cli(
            "--startup-report",
            self.report,
            "--standardized-output",
            self.report,
            "--physical-output",
            self.report,
            "--metadata",
            self.report,
            startup_only=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["construct", "self_test:"])
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})

    def test_new_report_keeps_legacy_field_overwrite_policy(self):
        self.output.write_text("previous fields\n")
        self.metadata.write_text("previous metadata\n")
        result = self.run_cli("--startup-report", self.report)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.output.read_text(), "published fields\n")
        self.assertEqual(self.metadata.read_text(), "published metadata\n")
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})

    def test_report_cannot_share_any_diagnostic_dump(self):
        for name in (
            "fx.f32",
            "embedding.f32",
            "permutation.i64",
            "source_cell_ids.i64",
        ):
            with self.subTest(name=name):
                directory = self.root / name.replace(".", "-")
                directory.mkdir()
                report = directory / name
                result = self.run_cli(
                    "--startup-report", report, "--dump-input-dir", directory
                )
                self.assert_report_rejected(result, "output paths must be distinct")
                self.assertEqual(list(directory.iterdir()), [])
                self.assertFalse(self.output.exists())

    def test_report_cannot_alias_dump_through_symlinked_directory(self):
        directory = self.root / "dumps"
        directory.mkdir()
        alias = self.root / "dump-alias"
        alias.symlink_to(directory, target_is_directory=True)
        result = self.run_cli(
            "--startup-report", alias / "fx.f32", "--dump-input-dir", directory
        )
        self.assert_report_rejected(result, "output paths must be distinct")
        self.assertEqual(list(directory.iterdir()), [])

    def test_separate_startup_report_survives_diagnostic_dumps(self):
        directory = self.root / "dumps"
        result = self.run_cli(
            "--startup-report", self.report, "--dump-input-dir", directory
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(self.report.read_text()), {"status": "passed"})
        self.assertEqual((directory / "fx.f32").read_text(), "diagnostic dump\n")

    def test_startup_only_report_can_match_inactive_diagnostic_dump(self):
        directory = self.root / "dumps"
        directory.mkdir()
        report = directory / "fx.f32"
        result = self.run_cli(
            "--startup-report", report, "--dump-input-dir", directory, startup_only=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded_events(), ["construct", "self_test:"])
        self.assertEqual(json.loads(report.read_text()), {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
