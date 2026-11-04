from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from qa.native_inference import cfd_contract, cfd_reference


@pytest.fixture
def local_assets(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (root / "qa/inference/cfd_parity_surface_run1_transolver.json").read_text()
    )
    request = json.loads((root / profile["rest"]["request_path"]).read_text())
    assets = {}
    for name, contents in (
        ("checkpoint", b"independent trusted weights"),
        ("statistics", b'{"mean": 1.0, "std": 2.0}'),
    ):
        pin = {
            "filename": cfd_contract.ASSET_PINS[name]["filename"],
            "sha256": hashlib.sha256(contents).hexdigest(),
            "size_bytes": len(contents),
        }
        monkeypatch.setitem(cfd_contract.ASSET_PINS, name, pin)
        profile["native_reference"][name] = pin
        path = tmp_path / pin["filename"]
        path.write_bytes(contents)
        assets[name] = {
            "path": str(path),
            "sha256": pin["sha256"],
            "size_bytes": pin["size_bytes"],
        }
    return profile, request, assets


def config(tmp_path, profile, request, assets):
    return cfd_reference.build_capture_config(
        profile,
        request,
        tmp_path / "inputs",
        tmp_path / "outputs",
        model_assets=assets,
    )


def test_reference_uses_explicit_verified_paths_before_provider_package_resolution(
    tmp_path, local_assets
):
    profile, request, assets = local_assets
    untouched = copy.deepcopy(profile)
    value = config(tmp_path, profile, request, assets)
    model = value["benchmark"]["models"][0]
    # The pinned provider returns these two explicit fields before creating its
    # remote Package, so a warm verified cache needs no HF metadata/token call.
    assert model.get("checkpoint") == assets["checkpoint"]["path"]
    assert model.get("stats_path") == assets["statistics"]["path"]
    assert model["package"] == cfd_contract.PACKAGE
    assert model["kwargs"] == untouched["config"]["models"][0]["kwargs"]
    assert value["run"]["seed"] == 42
    assert value["run"]["metrics_cache"] == {"enabled": False}
    assert value["benchmark"]["datasets"][0]["case_ids"] == ["run_1"]
    assert profile == untouched


@pytest.mark.parametrize("change", ["bytes", "digest", "partial", "filename"])
def test_reference_local_paths_cannot_bypass_exact_asset_pins(
    tmp_path, local_assets, change
):
    profile, request, assets = local_assets
    if change == "bytes":
        Path(assets["checkpoint"]["path"]).write_bytes(b"changed")
    elif change == "digest":
        assets["checkpoint"]["sha256"] = "0" * 64
    elif change == "partial":
        assets.pop("statistics")
    else:
        old = Path(assets["checkpoint"]["path"])
        other = old.with_name("Transolver.0.999.mdlus")
        other.write_bytes(old.read_bytes())
        assets["checkpoint"]["path"] = str(other)
    with pytest.raises(cfd_contract.FieldContractError):
        config(tmp_path, profile, request, assets)


def test_final_http_evaluator_reuses_verified_captured_paths(tmp_path, local_assets):
    _, _, assets = local_assets
    reference = tmp_path / "fields.json"
    reference.write_text(json.dumps({"local_model_assets": assets}))
    assert cfd_reference.reference_model_asset_arguments(reference) == [
        "--checkpoint",
        assets["checkpoint"]["path"],
        "--stats",
        assets["statistics"]["path"],
    ]
    Path(assets["statistics"]["path"]).write_bytes(b"changed")
    with pytest.raises(cfd_contract.FieldContractError):
        cfd_reference.reference_model_asset_arguments(reference)


def test_capture_records_actual_verified_asset_paths(
    tmp_path, local_assets, monkeypatch
):
    profile, request, assets = local_assets
    observed = []

    class Wrapper:
        _batch_resolution = 60000
        _inference_mode = "surface"
        _cuda_bf16_autocast = False

        def load(self, checkpoint_path, stats_path, device, **kwargs):
            observed.append((checkpoint_path, stats_path, device))
            return self

        def prepare_inputs(self, case):
            raise AssertionError("no inference needed to verify asset capture")

        predict = decode_outputs = prepare_inputs

    module_name = "physicsnemo.cfd.evaluation.models.wrappers.transolver.wrapper"
    provider = ModuleType(module_name)
    provider.TransolverWrapper = Wrapper
    monkeypatch.setitem(sys.modules, module_name, provider)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    observer = cfd_reference.ProviderCapture(
        profile=profile,
        request=request,
        root=tmp_path,
        image="registry.test/provider@sha256:" + "b" * 64,
        source_sha="a" * 40,
    )
    with observer.observe():
        Wrapper().load(
            assets["checkpoint"]["path"], assets["statistics"]["path"], "cuda:0"
        )
    assert observer.manifest["local_model_assets"] == assets
    assert observed == [
        (assets["checkpoint"]["path"], assets["statistics"]["path"], "cuda:0")
    ]


def test_named_checkpoint_symlink_preserves_pinned_epoch_filename(
    tmp_path, local_assets
):
    _, _, assets = local_assets
    checkpoint = Path(assets["checkpoint"]["path"])
    blob = checkpoint.with_name("content-addressed-blob")
    checkpoint.rename(blob)
    checkpoint.symlink_to(blob.name)
    assert cfd_reference.verify_local_model_assets(assets) == assets


def test_local_asset_flags_are_paired_and_legacy_reference_remains_supported(tmp_path):
    with pytest.raises(cfd_contract.FieldContractError, match="together"):
        cfd_reference.local_model_assets(tmp_path / "weights", None)
    assert cfd_reference.local_model_assets(None, None) is None
    reference = tmp_path / "legacy-fields.json"
    reference.write_text("{}")
    assert cfd_reference.reference_model_asset_arguments(reference) == []
