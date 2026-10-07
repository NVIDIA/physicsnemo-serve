"""Build-time image identity checks require matching VTK and a full source SHA."""

from pathlib import Path

import pytest

from qa.native_inference.image_metadata import validate_identity


def _identity(tmp_path, **changes):
    arguments = {
        "source_sha": "a" * 40,
        "native_vtk": "9.1.0",
        "python_vtk": "9.1.0",
        "python_vtk_path": tmp_path / "distro/vtkmodules/vtkCommonCore.so",
        "distro_python_root": tmp_path / "distro",
    }
    arguments.update(changes)
    return validate_identity(**arguments)


def test_matched_distro_vtk_records_full_source_identity(tmp_path):
    identity = _identity(tmp_path)
    assert identity["source_sha"] == "a" * 40
    assert identity["vtk_version"] == "9.1.0"


@pytest.mark.parametrize("source_sha", ["", "abcdef0", "g" * 40, "a" * 41])
def test_invalid_source_sha_cannot_label_an_image(tmp_path, source_sha):
    with pytest.raises(ValueError, match="full.*SHA"):
        _identity(tmp_path, source_sha=source_sha)


def test_mixed_native_python_vtk_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="VTK.*match"):
        _identity(tmp_path, python_vtk="9.6.2")


def test_wheel_shadowing_distro_vtk_is_rejected_even_with_same_version(tmp_path):
    with pytest.raises(ValueError, match="distro"):
        _identity(
            tmp_path, python_vtk_path=Path("/unrelated/vtkmodules/vtkCommonCore.so")
        )
