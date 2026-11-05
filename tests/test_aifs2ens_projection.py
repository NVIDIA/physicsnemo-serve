# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading a model must preserve Earth2Studio's native noise implementation."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/earth2-aifs2ens"))
import aifs2ens_runtime as runtime  # noqa: E402


@pytest.mark.parametrize("workspace", [None, ":4096:8"])
def test_loading_preserves_the_native_gpu_noise_projector(
    tmp_path, monkeypatch, workspace
):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    monkeypatch.setattr(
        runtime, "prepare_assets", lambda contract: tmp_path / "model.ckpt"
    )
    verified = []
    monkeypatch.setattr(
        runtime,
        "verify_model_contract",
        lambda model, contract: verified.append((model, contract)),
    )

    class Matrix:
        dtype = np.dtype("float32")
        device = "cpu"  # Native forward lazily moves this unregistered matrix.

        def to(self, device):
            assert device == "cuda", "the plugin moved native noise projection to CPU"
            return self

    class Projector:
        autocast = False
        projection_matrix = Matrix()

        def forward(self, values):
            return values

    projector = Projector()
    original_forward = projector.forward.__func__
    moves = []
    model = SimpleNamespace(seed=None, modules=lambda: [projector])
    model.to = lambda device: moves.append(device) or model
    model.eval = lambda: model
    calls = []

    def load(package, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True), float32=np.dtype("float32")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.auto",
        SimpleNamespace(Package=lambda path, **kwargs: SimpleNamespace(path=path)),
    )
    monkeypatch.setitem(
        sys.modules,
        "earth2studio.models.px",
        SimpleNamespace(AIFS2ENS=SimpleNamespace(load_model=load)),
    )
    monkeypatch.setitem(
        sys.modules,
        "anemoi.models.layers.sparse_projector",
        SimpleNamespace(SparseProjector=Projector),
    )
    contract = object()
    _, loaded = runtime.load_model(contract, "cuda")
    assert loaded is model and moves == ["cuda"]
    assert calls == [{"preload_invariants": False}]
    assert verified == [(model, contract)]
    assert projector.projection_matrix.device == "cpu"
    assert projector.forward.__func__ is original_forward
