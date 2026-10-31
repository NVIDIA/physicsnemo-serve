from __future__ import annotations

import copy
from pathlib import Path

import pytest

from qa.native_inference import cfd_overlay_guard as guard


def fixture():
    before = {
        "python": "3.12",
        "system": "Linux",
        "machine": "x86_64",
        "torch_cuda": "13.1",
        "protected": {
            "torch": {"version": "2.10.0", "files": [{"sha256": "real-base-hash"}]}
        },
        "packages": {"torch": {"version": "2.10.0", "requires": []}},
    }
    after = copy.deepcopy(before)
    after["protected"][guard.PROVIDER] = {
        "version": "0.0.2",
        "files": [{"sha256": "new-provider"}],
    }
    after["packages"].update(
        {
            name: {"version": version, "requires": []}
            for name, version in guard.OVERLAY_VERSIONS.items()
        }
    )
    contract = {
        key: before[key] for key in ("python", "system", "machine", "torch_cuda")
    }
    contract.update(versions={"torch": "2.10.0"}, metadata_exceptions=[])
    native = guard.load_native_guard(
        Path(__file__).resolve().parents[1]
        / "native-inference/model-builder/images/verify_builder_environment.py"
    )
    return before, after, contract, native


def test_only_pinned_cfd_addition_is_allowed_without_changing_generic_builder_guard():
    before, after, contract, native = fixture()
    with pytest.raises(ValueError, match="protected NGC stack changed"):
        native.verify_environment(before, after, contract)
    result = guard.verify_overlay(before, after, contract, native_guard=native)
    assert result["status"] == "passed"
    assert result["protected"] == before["protected"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda after: after["protected"]["torch"]["files"][0].update(
            sha256="replaced-library"
        ),
        lambda after: after["protected"].update(
            {"nvidia-unexpected": {"version": "1.0"}}
        ),
        lambda after: after["packages"]["pyvista"].update(version="0.46.0"),
        lambda after: after["protected"][guard.PROVIDER].update(version="0.0.3"),
    ],
)
def test_torch_mutation_unexpected_nvidia_and_wrong_overlay_versions_fail(mutation):
    before, after, contract, native = fixture()
    mutation(after)
    with pytest.raises(ValueError):
        guard.verify_overlay(before, after, contract, native_guard=native)
