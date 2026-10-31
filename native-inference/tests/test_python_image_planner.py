# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_image_cargo_planner_can_parse_native_workspace_member(tmp_path):
    import re
    import shutil

    dockerfile = (
        REPO_ROOT / "Dockerfile.physicsnemo-serve.scicomp-rust-slim"
    ).read_text()
    planner = dockerfile.split("FROM chef AS planner", 1)[1].split(
        "FROM chef AS builder", 1
    )[0]
    for line in planner.splitlines():
        if line.startswith("COPY "):
            *sources, destination = line.split()[1:]
            target = tmp_path / destination
            target.mkdir(parents=True, exist_ok=True)
            for source in sources:
                shutil.copy2(REPO_ROOT / source, target / Path(source).name)
    stub = re.search(r"RUN mkdir .*?(?=\n\n)", planner, re.DOTALL)
    assert stub
    subprocess.run(["bash", "-ec", stub.group()[4:]], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--offline", "--format-version=1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
