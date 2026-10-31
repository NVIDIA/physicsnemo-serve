"""Resolve the lower-level CFD image's source and installed-library closure."""

from pathlib import Path, PurePosixPath
import re
import shlex


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "native-inference/deploy/Dockerfile.native"


def instructions():
    recipe = RECIPE.read_text().replace("\\\n", " ")
    return [
        line.split(None, 1)
        for line in recipe.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def environment(stage):
    return dict(
        value.split("=", 1)
        for instruction, command in stage
        if instruction == "ENV"
        for value in shlex.split(command)
    )


def test_native_image_local_copy_sources_exist():
    missing = [
        source
        for instruction, command in instructions()
        if instruction == "COPY" and not command.startswith("--from=")
        for source in shlex.split(command)[:-1]
        if not (ROOT / source).exists()
    ]
    assert not missing, f"Docker COPY sources missing from release checkout: {missing}"


def test_native_image_builds_and_carries_the_worker_workflow_host():
    recipe = instructions()
    stages = [index for index, (op, _) in enumerate(recipe) if op == "FROM"]
    build = recipe[stages[0] : stages[1]]
    release = recipe[stages[1] :]
    cargo = next(
        index
        for index, (op, command) in enumerate(build)
        if op == "RUN" and "cargo build" in command
    )
    build = build[:cargo]
    build_env = environment(build)
    include = build_env.get("NATIVE_WORKFLOW_INCLUDE")
    lib = build_env.get("NATIVE_WORKFLOW_LIB_DIR")
    assert include and lib, (
        "native-worker cpp requires installed workflow headers/library"
    )
    prefix = str(PurePosixPath(include).parent)
    assert PurePosixPath(lib).parent == PurePosixPath(prefix)

    # Follow the configured install to its copied source, rather than requiring a
    # particular temporary build directory or spelling of the COPY instruction.
    configured = []
    for index, (op, command) in enumerate(build):
        if op != "RUN":
            continue
        for match in re.finditer(r"cmake -S (\S+) -B (\S+)(.*?)(?=&&|$)", command):
            options = shlex.split(match[3])
            if f"-DCMAKE_INSTALL_PREFIX={prefix}" in options:
                configured.append((index, match[1], match[2], command))
    assert len(configured) == 1, "workflow install directory has no CMake producer"
    configure, source, build_dir, command = configured[0]
    assert f"cmake --build {build_dir}" in command
    assert f"cmake --install {build_dir}" in command
    copied_sources = [
        ROOT / fields[0]
        for op, value in build[:configure]
        if op == "COPY" and not value.startswith("--from=")
        if (fields := shlex.split(value))[-1].rstrip("/") == source.rstrip("/")
    ]
    assert any(
        (path / "include/physicsnemo/workflow/v1/host.hpp").is_file()
        for path in copied_sources
    ), "workflow CMake source must include the header consumed by native-worker"

    copies = [shlex.split(value) for op, value in release if op == "COPY"]
    assert any(
        fields == ["--from=native-build", prefix, prefix] for fields in copies
    ), "workflow library must survive the final stage"
    for stage in (build, release):
        assert lib in environment(stage).get("LD_LIBRARY_PATH", "").split(":"), (
            "workflow library must be discoverable while linking and serving"
        )
