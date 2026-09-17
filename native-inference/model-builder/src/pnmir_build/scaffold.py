"""Create a small, framework-free model authoring project without overwriting files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

from .authoring_config import validate_source
from .projects import _absolute


_ADAPTER = '''"""Connect your existing model and representative inputs to Model Builder.

Add the files/packages imported below to "source" in model-build.json.
Set "checkpoint" to your trained plain tensor state_dict. The builder loads it.
Use "config" for constructor settings and "assets" for validation input files.
"""


def create_model(config, assets):
    """Construct the architecture on CPU; the builder loads the selected weights."""
    # Example: from model import MyModel
    #          return MyModel(**config).eval()
    raise NotImplementedError(
        "Edit build_adapter.py:create_model to construct your existing Python model."
    )


def create_cases(config, assets):
    """Return a list of positional input tuples from your validation data.

    For model(features, context), return [(features, context), ...].
    Use preprocessed float32 tensors with the same static shapes in every case.
    The builder computes Python reference outputs and checks native parity.
    """
    # Example: import torch
    #          return torch.load(assets["validation_inputs"],
    #                            map_location="cpu", weights_only=True)
    raise NotImplementedError(
        "Edit build_adapter.py:create_cases to return representative model inputs."
    )
'''


def _exclusive_write(path: Path, text: str) -> tuple[int, int]:
    """Publish a complete file with an atomic, no-replacement hard link."""
    fd, temporary = tempfile.mkstemp(prefix=".model-builder-init-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        identity = temporary_path.stat()
        os.link(temporary_path, path)
        return identity.st_dev, identity.st_ino
    finally:
        temporary_path.unlink(missing_ok=True)


def initialize(
    directory: Path,
    checkpoint: Path | None = None,
    source: list[str] | None = None,
) -> dict:
    """Create only the project and adapter, leaving model and environment setup open."""
    root = Path(directory).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("The initialization directory must not be a symlink.")
    root = root.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"The initialization destination is not a directory: {root}")
    targets = [root / "model-build.json", root / "build_adapter.py"]
    for target in targets:
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to overwrite existing file: {target}")
    if source is None:
        model = root / "model.py"
        source = ["model.py"] if model.is_file() and not model.is_symlink() else []
    validate_source(source, root)
    name = re.sub(r"[^a-z0-9_.-]+", "-", root.name.lower()).strip("-._")
    selected_checkpoint = None
    if checkpoint is not None:
        selected = _absolute(checkpoint)
        selected_checkpoint = str(
            selected.relative_to(root) if selected.is_relative_to(root) else selected
        )
    document = {
        "format_version": 2,
        "name": name or "my-model",
        "version": "0.1.0",
        "adapter": "build_adapter.py",
        "source": list(source),
        "checkpoint": selected_checkpoint,
        "config": {},
        "assets": {},
        "backends": ["aoti"],
        "executor": "container",
        "builder_image": None,
        "device": "cuda",
        "output_root": "builds",
    }
    contents = [json.dumps(document, indent=2, allow_nan=False) + "\n", _ADAPTER]
    created = []
    root.mkdir(parents=True, exist_ok=True)
    try:
        for target, content in zip(targets, contents):
            identity = _exclusive_write(target, content)
            created.append((target, identity))
    except OSError as exc:
        # A competing process may have replaced a file after publication. Remove
        # only the inode published by this call, never a pre-existing replacement.
        for target, identity in reversed(created):
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise ValueError(f"Cannot initialize model project {root}: {exc}") from exc
    return {
        "project": str(targets[0]),
        "created": [str(target) for target in targets],
        "next_steps": [
            "Set checkpoint, config, source and optional assets in model-build.json.",
            "Connect your model and validation inputs in build_adapter.py.",
            "Select builder_image or a compatible local environment in model-build.json.",
            f"Run physicsnemo-model-builder check {root}.",
            f"Run physicsnemo-model-builder build {root}.",
        ],
    }
