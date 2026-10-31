from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "native-inference/deploy"


def test_portable_http_controller_import_does_not_require_posix_fcntl(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['fcntl'] = None; import native_release_http_job",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(DEPLOY), str(ROOT))),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def qa_trees(tmp_path):
    current, stale = tmp_path / "sealed-source", tmp_path / "old-base"
    for directory in ("native_inference", "scripts"):
        shutil.copytree(
            ROOT / "qa" / directory,
            current / "qa" / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    old = stale / "qa/scripts/run_native_cfd_reference.py"
    old.parent.mkdir(parents=True)
    old.write_text(
        "import argparse\n"
        "def parser():\n"
        " p=argparse.ArgumentParser()\n"
        " e=p.add_subparsers(dest='command',required=True).add_parser('evaluate')\n"
        " for name in ('profile','request','input-root','image','source-sha','fields','output'):\n"
        "  e.add_argument('--'+name,required=True)\n"
        " return p\n"
    )
    return current, stale


def release_environment(current, stale):
    # Apply the final image's actual override to the inherited base-image value.
    # The subprocess below exercises imports and argparse, not just this text.
    release = (
        (DEPLOY / "Dockerfile.workflow")
        .read_text()
        .split("FROM ${NATIVE_SERVER_IMAGE} AS release", 1)[1]
    )
    overrides = re.findall(r'^ENV PYTHONPATH="([^"]+)"$', release, flags=re.M)
    paths = overrides[-1].split(":") if overrides else [str(stale)]
    value = os.pathsep.join(
        path.replace("/opt/native-release/source", str(current)).replace(
            "/opt/native-release", str(DEPLOY)
        )
        for path in paths
    )
    return {**os.environ, "PYTHONPATH": value}


def test_final_image_child_import_and_parser_use_sealed_qa_over_stale_base(tmp_path):
    current, stale = qa_trees(tmp_path)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from qa.scripts import run_native_cfd_reference as r; "
            "a=r.parser().parse_args(['evaluate','--profile','profile.json',"
            "'--request','request.json','--input-root','inputs','--image','image',"
            "'--source-sha','source','--fields','fields.json','--output','out',"
            "'--checkpoint','Transolver.0.501.mdlus','--stats','global_stats.json']);"
            "print(json.dumps({'origin':r.__file__,'checkpoint':str(a.checkpoint),'stats':str(a.stats)}))",
        ],
        cwd=tmp_path,
        env=release_environment(current, stale),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert Path(result["origin"]).is_relative_to(current)
    assert result["checkpoint"] == "Transolver.0.501.mdlus"
    assert result["stats"] == "global_stats.json"


@pytest.mark.parametrize("selected", ["current", "stale"])
def test_cpu_candidate_probe_checks_actual_module_origins_hashes_and_capability(
    tmp_path, selected
):
    current, stale = qa_trees(tmp_path)
    environment = release_environment(current, stale)
    if selected == "stale":
        environment["PYTHONPATH"] = os.pathsep.join((str(DEPLOY), str(stale)))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from native_release import inspect_qa_sources; "
            f"print(json.dumps(inspect_qa_sources({str(current / 'qa')!r})))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if selected == "stale":
        assert child.returncode != 0
        assert "QA module is outside sealed source" in child.stderr
    else:
        assert child.returncode == 0, child.stdout + child.stderr
        evidence = json.loads(child.stdout)
        assert evidence["local_model_asset_flags"] is True
        assert len(evidence["modules"]) == 9
        for record in evidence["modules"].values():
            path = Path(record["path"])
            assert path.is_relative_to(current)
            assert record["size_bytes"] == path.stat().st_size
            with path.open("rb") as source:
                assert (
                    record["sha256"]
                    == hashlib.file_digest(source, "sha256").hexdigest()
                )
