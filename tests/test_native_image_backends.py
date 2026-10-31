"""Host-only checks for Docker's backend options reaching native consumers.

These compile a small consumer using the options from the actual image recipe.
They validate build selection, not CUDA execution or TensorRT correctness.
"""

from pathlib import Path
import re
import shlex
from string import Template
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "native-inference" / "deploy"


def configured_backends(tmp_path, image, source, prefix, backend, *, exact=False):
    recipe = (DEPLOY / image).read_text()
    environment = dict(re.findall(r"^ARG (\w+)=(.*)$", recipe, re.MULTILINE))
    environment.update(
        ENABLE_AOTI="ON" if backend == "aoti" else "OFF",
        ENABLE_TENSORRT="ON" if backend == "tensorrt" else "OFF",
    )
    command = re.search(
        rf"cmake -S {re.escape(source)} -B .*?(?=&& cmake --build)",
        recipe,
        re.DOTALL,
    )
    assert command is not None, f"missing configure step for {source}"
    arguments = shlex.split(
        Template(command.group()).safe_substitute(environment).replace("\\\n", " ")
    )
    options = [
        argument
        for argument in arguments
        if argument.startswith(
            (f"-D{prefix}_ENABLE_AOTI=", f"-D{prefix}_ENABLE_TENSORRT=")
        )
    ]
    assert len(options) == 2, "each image must select both backend options explicitly"
    options.extend(
        argument
        for argument in arguments
        if argument.startswith("-DPNMIR_ENABLE_TENSORRT_EXACT=")
    )
    source_dir = tmp_path / "consumer"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(ImageBackendConsumer LANGUAGES CXX)\n"
        "add_executable(consumer main.cpp)\n"
        f"if({prefix}_ENABLE_AOTI)\n"
        "  target_compile_definitions(consumer PRIVATE HAS_AOTI=1)\n"
        "endif()\n"
        f"if({prefix}_ENABLE_TENSORRT)\n"
        "  target_compile_definitions(consumer PRIVATE HAS_TENSORRT=1)\n"
        "endif()\n"
        "if(PNMIR_ENABLE_TENSORRT_EXACT)\n"
        "  target_compile_definitions(consumer PRIVATE HAS_EXACT=1)\n"
        "endif()\n"
    )
    (source_dir / "main.cpp").write_text(
        "#include <iostream>\nint main(int argc, char**) {\n"
        'if (argc > 1) {\n#ifdef HAS_EXACT\nstd::cout << "enabled\\n";\n'
        '#else\nstd::cout << "disabled\\n";\n#endif\nreturn 0;\n}\n'
        '#ifdef HAS_AOTI\nstd::cout << "aoti\\n";\n#endif\n'
        '#ifdef HAS_TENSORRT\nstd::cout << "tensorrt\\n";\n#endif\n}\n'
    )
    build = tmp_path / "build"
    for argv in (
        ["cmake", "-S", str(source_dir), "-B", str(build), *options],
        ["cmake", "--build", str(build)],
    ):
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(build / "consumer"), *(["--exact"] if exact else [])],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


@pytest.mark.parametrize("backend", ["aoti", "tensorrt"])
@pytest.mark.parametrize(
    "image,source,prefix",
    [
        (
            "Dockerfile.workflow",
            "native-inference/workflows/transolver",
            "PNMIR",
        ),
        ("Dockerfile.workflow", "native-inference/workflows/runtime", "PNMS"),
        (
            "Dockerfile.runner",
            "/opt/physicsnemo/native-inference/cpp-runtime",
            "PNMIR",
        ),
        (
            "Dockerfile.runner",
            "/opt/physicsnemo/native-inference/workflows/transolver",
            "PNMIR",
        ),
    ],
)
def test_native_image_builds_the_selected_backend(
    tmp_path, image, source, prefix, backend
):
    assert configured_backends(tmp_path, image, source, prefix, backend) == [backend]


def test_manual_runner_preserves_exact_tensorrt_plugin_support(tmp_path):
    assert configured_backends(
        tmp_path,
        "Dockerfile.runner",
        "/opt/physicsnemo/native-inference/cpp-runtime",
        "PNMIR",
        "tensorrt",
        exact=True,
    ) == ["enabled"]
