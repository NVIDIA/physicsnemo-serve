# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate, fetch, and stage immutable inputs for a complete ensemble."""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Annotated, Callable, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

MAX_SEED = 2**63 - 1
NUMERICAL_POLICY = "earth2studio-native-v1"
RNG_POLICY = "torch-seed-once-native-v1"
MemberId = Annotated[StrictInt, Field(ge=1, le=50)]
Variable = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_time: str
    ifs_member_id: MemberId | None = None
    nsteps: Annotated[StrictInt, Field(ge=1, le=60)]
    nensemble: Annotated[StrictInt, Field(ge=1, le=16)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=16)] | None = None
    seed_base: Annotated[StrictInt, Field(ge=0, le=MAX_SEED)] = 1000
    output_variables: list[Variable] = Field(
        default_factory=lambda: ["t2m", "msl", "u10m", "v10m"], min_length=1
    )

    @field_validator("start_time")
    @classmethod
    def utc_cycle(cls, value: str) -> str:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("start_time must include a UTC offset")
        date = date.astimezone(timezone.utc)
        if date.hour % 6 or date.minute or date.second or date.microsecond:
            raise ValueError("start_time must be a six-hour UTC initialization cycle")
        return date.isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def ensemble_contract(self):
        self.batch_size = min(self.nensemble, self.batch_size or self.nensemble)
        if len(set(self.output_variables)) != len(self.output_variables):
            raise ValueError("output_variables must be unique")
        return self


EARTH2STUDIO_REVISION = "f8af4030cfd43ec79bf4c0cf6c23463a87bfef97"
CHECKPOINT_REVISION = "29a1f8c4ab76cd0e5b64cfd415f4ff9e08edf7cf"
INVARIANTS = ("lsm", "sdor", "slor", "z", "wmb")
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class InputContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    checkpoint_sha256: Digest
    invariants_sha256: Digest | None = None
    interpolation_sha256: Digest | None = None
    inverse_interpolation_sha256: Digest | None = None
    earth2studio_revision: Literal[EARTH2STUDIO_REVISION] = EARTH2STUDIO_REVISION
    checkpoint_revision: Literal[CHECKPOINT_REVISION] = CHECKPOINT_REVISION
    input_variables: list[str] = Field(min_length=1)
    output_variables: list[str] = Field(min_length=1)
    lat: list[float] = Field(min_length=1)
    lon: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def coordinate_contract(self):
        for names in (self.input_variables, self.output_variables):
            if len(names) != len(set(names)):
                raise ValueError("contract variable names must be unique")
        if not set(INVARIANTS).issubset(self.input_variables):
            raise ValueError("contract must include all five invariant inputs")
        for values in (self.lat, self.lon):
            if not np.isfinite(values).all() or len(values) != len(set(values)):
                raise ValueError("contract coordinates must be finite and unique")
        if np.any(np.diff(self.lat) >= 0) or np.any(np.diff(self.lon) <= 0):
            raise ValueError("expected descending latitude and ascending longitude")
        return self

    def validate_values(self, values):
        expected = (1, 2, len(self.input_variables), len(self.lat), len(self.lon))
        if values.shape != expected:
            raise ValueError(f"input shape {values.shape} differs from {expected}")
        if values.dtype != np.dtype("<f4"):
            raise ValueError("staged inputs must be little-endian float32")
        for name in INVARIANTS:
            index = self.input_variables.index(name)
            if not np.array_equal(
                values[:, 0, index], values[:, 1, index], equal_nan=True
            ):
                raise ValueError(f"invariant {name} differs between input times")

    def coords(self, request: ForecastRequest):
        return OrderedDict(
            time=np.array(
                [request.start_time.removesuffix("Z")], dtype="datetime64[ns]"
            ),
            lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
            variable=np.array(self.input_variables),
            lat=np.array(self.lat),
            lon=np.array(self.lon),
        )


@dataclass
class WeatherInput:
    values: np.ndarray
    provenance: dict


@dataclass
class LoadedEnsembleInput:
    values: np.ndarray
    coords: OrderedDict
    request: ForecastRequest
    contract: InputContract
    provenance: dict
    manifest_sha256: str = ""
    input_sha256: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _check_digest(path: Path, expected: str):
    if sha256_file(path) != expected:
        raise ValueError(f"artifact digest mismatch: {path.name}")


def _relative_artifact(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its staged input directory")
    return path


def stage_inputs(
    request: ForecastRequest,
    contract: InputContract,
    run_dir: Path,
    fetch: Callable[[ForecastRequest, InputContract], WeatherInput],
    *,
    max_output_bytes: int,
) -> dict:
    unknown = set(request.output_variables) - set(contract.output_variables)
    if unknown:
        raise ValueError(f"unsupported output variable(s): {sorted(unknown)}")
    output_bytes = (
        request.nensemble
        * request.nsteps
        * len(request.output_variables)
        * len(contract.lat)
        * len(contract.lon)
        * 4
    )
    if max_output_bytes <= 0 or output_bytes > max_output_bytes:
        raise ValueError(
            f"estimated output {output_bytes} exceeds output storage budget"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="staged-inputs-", dir=run_dir)).resolve()
    try:
        contract_path = root / "input-contract.json"
        write_json(contract_path, contract.model_dump(mode="json"))
        fetched = fetch(request, contract)
        contract.validate_values(fetched.values)
        target = root / "input.npy"
        np.save(target, fetched.values, allow_pickle=False)
        manifest = {
            "schema_version": 2,
            "request": request.model_dump(mode="json"),
            "contract_path": contract_path.name,
            "contract_sha256": sha256_file(contract_path),
            "input_path": target.name,
            "input_sha256": sha256_file(target),
            "provenance": fetched.provenance,
            "estimated_output_bytes": output_bytes,
        }
        manifest_path = root / "manifest.json"
        write_json(manifest_path, manifest)
        return {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    except BaseException:
        shutil.rmtree(root)
        raise


def load_ensemble_input(manifest_path: str, expected_sha256: str):
    path = Path(manifest_path).resolve()
    _check_digest(path, expected_sha256)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported staged input schema")
    request = ForecastRequest.model_validate(manifest["request"])
    contract_path = _relative_artifact(path.parent, manifest["contract_path"])
    _check_digest(contract_path, manifest["contract_sha256"])
    contract = InputContract.model_validate_json(contract_path.read_text())
    target = _relative_artifact(path.parent, manifest["input_path"])
    _check_digest(target, manifest["input_sha256"])
    values = np.load(target, allow_pickle=False, mmap_mode="r")
    contract.validate_values(values)
    return LoadedEnsembleInput(
        values,
        contract.coords(request),
        request,
        contract,
        manifest["provenance"],
        expected_sha256,
        manifest["input_sha256"],
    )


PREPROCESSING_POLICY = "earth2studio-latlon-frozen-invariants-v1"


class WeatherSource:
    def __init__(self, fetch=None):
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(request, variables):
        from aifs2ens_runtime import _verify_runtime

        _verify_runtime()
        from earth2studio.data import IFS, IFS_ENS, fetch_data
        from plugin_sdk import cleanup_earth2_runtime_resources

        mirror = os.environ.get("AIFS2ENS_SOURCE", "ecmwf")
        if mirror not in {"ecmwf", "aws", "google", "azure"}:
            raise ValueError("unsupported AIFS2ENS_SOURCE mirror")
        source = (
            IFS(source=mirror, verbose=False)
            if request.ifs_member_id is None
            else IFS_ENS(source=mirror, member=request.ifs_member_id, verbose=False)
        )
        try:
            values, coords = fetch_data(
                source=source,
                time=np.array(
                    [request.start_time.removesuffix("Z")], dtype="datetime64[ns]"
                ),
                variable=np.array(variables),
                lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
                device="cpu",
            )
            return values.detach().cpu().numpy(), coords
        finally:
            cleanup_earth2_runtime_resources(source)

    def __call__(self, request, contract):
        configured = os.environ.get("AIFS2ENS_INVARIANTS")
        if not configured or contract.invariants_sha256 is None:
            raise ValueError("qualified invariants path and digest are required")
        path = Path(configured)
        digest = sha256_file(path)
        if digest != contract.invariants_sha256:
            raise ValueError("invariant artifact digest differs from contract")
        fixed = np.load(path, allow_pickle=False)
        height, width = len(contract.lat), len(contract.lon)
        if fixed.shape != (5, height, width) or fixed.dtype != np.dtype("<f4"):
            raise ValueError("invariant shape/dtype differs from model contract")
        variables = [v for v in contract.input_variables if v not in INVARIANTS]
        values, coords = self.fetch(request, variables)
        expected = contract.coords(request)
        expected["variable"] = np.array(variables)
        for name, coordinate in expected.items():
            if name not in coords or not np.array_equal(coords[name], coordinate):
                raise ValueError(
                    f"weather {name} coordinates differ from model contract"
                )
        if values.shape != (
            1,
            2,
            len(variables),
            height,
            width,
        ) or values.dtype != np.dtype("<f4"):
            raise ValueError("weather input shape/dtype differs from model contract")
        fields = {name: values[:, :, i : i + 1] for i, name in enumerate(variables)}
        for index, name in enumerate(INVARIANTS):
            fields[name] = np.broadcast_to(
                fixed[index][None, None, None], (1, 2, 1, height, width)
            )
        combined = np.concatenate([fields[v] for v in contract.input_variables], axis=2)
        return WeatherInput(
            combined,
            {
                "earth2studio_revision": EARTH2STUDIO_REVISION,
                "initialization": request.start_time,
                "ifs_member_id": request.ifs_member_id,
                "mirror": os.environ.get("AIFS2ENS_SOURCE", "ecmwf"),
                "invariants_sha256": digest,
                "preprocessing_policy": PREPROCESSING_POLICY,
            },
        )
