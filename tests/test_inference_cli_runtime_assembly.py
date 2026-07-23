# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = REPO_ROOT / "packaging" / "physicsnemo-serve-cmd" / "assemble_runtime.py"


def _load_assembler():
    spec = importlib.util.spec_from_file_location(
        "physicsnemo_serve_runtime_assembler", ASSEMBLER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assemble_runtime_copies_interpreter_and_runner_support(tmp_path: Path) -> None:
    module = _load_assembler()
    python_prefix = tmp_path / "python-prefix"
    python = python_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f'#!/bin/sh\nexec {sys.executable!r} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    output = tmp_path / "runtime"

    module.assemble_runtime(
        repo_root=REPO_ROOT,
        python_prefix=python_prefix,
        output=output,
        requirements=None,
        wheelhouse=None,
    )

    assert (output / "bin" / "python").is_file()
    assert (output / "scripts" / "plugin_direct_runner.py").is_file()
    assert (output / "scripts" / "plugin_runtime.py").is_file()
    assert (output / "scripts" / "plugin_sdk.py").is_file()
    assert (output / "python").is_dir()
    manifest = json.loads(
        (output / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["requirements_sha256"] is None


def test_assemble_runtime_rejects_missing_interpreter(tmp_path: Path) -> None:
    module = _load_assembler()

    try:
        module.assemble_runtime(
            repo_root=REPO_ROOT,
            python_prefix=tmp_path / "missing",
            output=tmp_path / "runtime",
            requirements=None,
            wheelhouse=None,
        )
    except ValueError as exc:
        assert "bin/python" in str(exc)
    else:
        raise AssertionError("missing interpreter should be rejected")
