"""Read-only project lock planning and atomic publication with conflict checks.

The short-lived sibling publication directory serializes cooperating writers.
A leftover directory after a crash fails closed; it is never silently stolen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile


def _path(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    return path.parent.resolve(strict=True) / path.name


def _hash(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def _json_value(value) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("project lock object keys must be strings")
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("project lock numbers must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("project lock identities must contain only JSON values")


def _encode(value: dict) -> bytes:
    _json_value(value)
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _key(value) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("project lock entry key must be a nonempty string")


def _validate(content: dict) -> None:
    if (
        type(content) is not dict
        or set(content) != {"format_version", "entries"}
        or type(content["format_version"]) is not int
        or content["format_version"] != 1
        or type(content["entries"]) is not dict
    ):
        raise ValueError("project lock requires format_version 1 and entries only")
    for key, identity in content["entries"].items():
        _key(key)
        if type(identity) is not dict:
            raise ValueError("project lock entries must contain identity objects")
    _json_value(content)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate project lock key: {key}")
        result[key] = value
    return result


def _decode(data: bytes) -> dict:
    content = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    _validate(content)
    return content


def _read(path: Path) -> bytes | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(
            "project lock must be a regular file, not a symlink or directory"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError("project lock changed while opening") from error
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("project lock changed while opening")
        return handle.read()


def inspect_lock(path: Path, key: str, identity: dict, *, update: bool = False) -> dict:
    """Return a detached prospective lock state without changing the filesystem."""
    _key(key)
    if type(identity) is not dict or type(update) is not bool:
        raise ValueError("project lock requires an identity object and boolean update")
    selected = _encode(identity)
    path = _path(path)
    original = _read(path)
    content = (
        _decode(original)
        if original is not None
        else {"format_version": 1, "entries": {}}
    )
    previous = content["entries"].get(key)
    if previous is not None and _encode(previous) != selected and not update:
        raise ValueError(
            f"project lock identity changed for {key}; explicit update required"
        )
    content["entries"][key] = json.loads(selected)
    return {
        "path": str(path),
        "original_sha256": _hash(original),
        "content": content,
        "content_sha256": _hash(_encode(content)),
    }


def publish_lock(path: Path, inspected: dict) -> dict:
    """Publish exactly the inspected state, rejecting intervening lock changes."""
    path = _path(path)
    if (
        type(inspected) is not dict
        or set(inspected) != {"path", "original_sha256", "content", "content_sha256"}
        or inspected["path"] != str(path)
    ):
        raise ValueError("project lock inspection does not match destination")
    _validate(inspected["content"])
    encoded = _encode(inspected["content"])
    if _hash(encoded) != inspected["content_sha256"]:
        raise ValueError("project lock inspection content changed")
    guard = path.with_name(f".{path.name}.publish")
    try:
        guard.mkdir()
    except FileExistsError as error:
        raise ValueError(
            "concurrent project lock publication; inspect again after it finishes"
        ) from error
    temporary = None
    try:
        original = _read(path)
        if _hash(original) != inspected["original_sha256"]:
            raise ValueError("project lock changed after inspection; inspect again")
        if original is not None and _encode(_decode(original)) == encoded:
            encoded = original  # Preserve customer formatting when nothing changed.
        else:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if _hash(_read(path)) != inspected["original_sha256"]:
                raise ValueError(
                    "project lock changed during publication; inspect again"
                )
            os.replace(temporary, path)
        return {
            "path": str(path),
            "sha256": _hash(encoded),
            "size_bytes": len(encoded),
            "serialized_text": encoded.decode("utf-8"),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        guard.rmdir()
