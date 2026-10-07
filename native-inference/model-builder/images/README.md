# GeoTransolver builder environment

Build `Dockerfile.builder` from the repository root. The default image includes
AOTI and the GeoTransolver surface-core producer dependencies. TensorRT remains
an optional SDK build argument; it is not a qualified GeoTransolver backend.
No checkpoint or generated model belongs in the image. Supply the checkpoint
through the model-builder CLI at execution time.
The image sets PhysicsNeMo's `LOCAL_CACHE` to a container-local temporary path;
the customer UID creates it at runtime without needing a home directory entry.

The Linux x86_64 NGC 26.01 base is pinned by platform digest.
`geotransolver-requirements.lock` adds 26 exact, SHA-256 checked PyPI artifacts;
`geotransolver-environment.json` pins all 75 packages in their active dependency
closure, including packages supplied by the base. The source-only ANTLR 4.9.3
artifact builds using the base's pinned setuptools with build isolation disabled.
The `builder-environment` Docker target builds just this dependency layer.

`verify_builder_environment.py` captures the base stack, verifies all active
requirements (including requested extras), and rejects changes to protected
Torch, torchvision, torchaudio, Triton, TensorRT, and NVIDIA distribution
metadata/native libraries. The two explicit metadata exceptions retain the exact
qualified NGC Torch/torchvision prereleases despite PhysicsNeMo's stable-version
requirements. Other missing or incompatible dependencies fail the build.
The image keeps its lock, contract, base snapshot, and verification receipt under
`/opt/physicsnemo/image/`.

Update a base or producer version by regenerating the complete closure in an
isolated candidate environment, obtaining artifact hashes from PyPI, updating
both lock and contract, and passing the environment guard plus actual GPU model
preparation/export/native full-reference qualification. The lock alone does not
qualify a new stack or promise byte-identical compiler outputs. Never resolve
Torch dependencies implicitly when updating this overlay.
