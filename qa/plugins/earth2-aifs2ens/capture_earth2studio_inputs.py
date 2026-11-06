# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently freeze stock Earth2Studio data with its supported invariant-input API."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

INVARIANTS = ("lsm", "sdor", "slor", "z", "wmb")
PREPROCESSING_POLICY = "earth2studio-latlon-frozen-invariants-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ifs-member-id", type=int, choices=range(1, 51), default=1)
    args = parser.parse_args()
    args.original_assets = args.original_assets.resolve()
    from earth2studio.data import IFS_ENS, fetch_data

    provenance = json.loads(
        importlib.metadata.distribution("earth2studio").read_text("direct_url.json")
    )
    if (
        provenance.get("vcs_info", {}).get("commit_id")
        != "f8af4030cfd43ec79bf4c0cf6c23463a87bfef97"
    ):
        raise ValueError("input capture requires the pinned Earth2Studio source")
    contract = json.loads(
        (args.original_assets / "qualified-input-contract.json").read_text()
    )
    previous = json.loads(
        (args.original_assets / "weather-capture-report.json").read_text()
    )
    args.output.mkdir(exist_ok=False)
    assets = args.output / "assets"
    assets.mkdir()
    for name in ("invariants.npy", "checkpoint-xet"):
        (assets / name).symlink_to(args.original_assets / "assets" / name)
    (args.output / "qualified-input-contract.json").write_bytes(
        (args.original_assets / "qualified-input-contract.json").read_bytes()
    )
    fixed_path = assets / "invariants.npy"
    if digest(fixed_path) != contract["invariants_sha256"]:
        raise ValueError("frozen invariant bytes changed")
    fixed = np.load(fixed_path, allow_pickle=False)
    variables = [name for name in contract["input_variables"] if name not in INVARIANTS]
    member = args.ifs_member_id
    print(
        f"Capturing unchanged Earth2Studio fields for IFS member {member}",
        flush=True,
    )
    source = IFS_ENS(member=member, source="ecmwf", verbose=False)
    values, coords = fetch_data(
        source=source,
        time=np.array(
            [previous["initialization"].removesuffix("Z")], dtype="datetime64[ns]"
        ),
        variable=np.array(variables),
        lead_time=np.array([-6, 0], dtype="timedelta64[h]"),
        device="cpu",
    )
    expected = {
        "time": np.array(
            [previous["initialization"].removesuffix("Z")], dtype="datetime64[ns]"
        ),
        "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
        "variable": np.array(variables),
        "lat": np.array(contract["lat"]),
        "lon": np.array(contract["lon"]),
    }
    for name, coordinate in expected.items():
        np.testing.assert_array_equal(coords[name], coordinate)
    values = values.cpu().numpy()
    if values.dtype != np.float32 or values.shape != (
        1,
        2,
        len(variables),
        721,
        1440,
    ):
        raise ValueError("Earth2Studio input shape/dtype changed")
    fields = {
        name: values[:, :, index : index + 1] for index, name in enumerate(variables)
    }
    for index, name in enumerate(INVARIANTS):
        fields[name] = np.broadcast_to(
            fixed[index][None, None, None], (1, 2, 1, 721, 1440)
        )
    combined = np.concatenate(
        [fields[name] for name in contract["input_variables"]], axis=2
    )
    target = assets / "input.npy"
    np.save(target, combined, allow_pickle=False)
    captured_input = {
        "path": str(target),
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "shape": list(combined.shape),
        "provenance": {
            "mirror": "ecmwf",
            "ifs_member_id": member,
            "preprocessing_policy": PREPROCESSING_POLICY,
        },
    }
    del values, fields, combined, source
    report = {
        "status": "passed",
        "initialization": previous["initialization"],
        "input": captured_input,
        "shared_input": True,
        "ifs_member_id": member,
        "preprocessing_policy": PREPROCESSING_POLICY,
        "invariants_sha256": contract["invariants_sha256"],
        "capture_script_sha256": digest(__file__),
        "model_input_mode": "preload_invariants=False",
    }
    (args.output / "weather-capture-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
