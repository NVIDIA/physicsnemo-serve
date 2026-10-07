"""Download and verify the pinned Transolver surface QA assets on Linux/macOS."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


if __package__:
    from .contract import sha256
else:
    from contract import sha256


MODEL_REVISION = "96477aeb86d24c26ccf0797bca1b3851268017d0"
DATA_REVISION = "5d448b209bf654503c64ce7261c34fa125f46392"
MODEL_URL = (
    "https://huggingface.co/nvidia/transolver_drivaerml/resolve/"
    f"{MODEL_REVISION}/transolver_drivaerml_surface_checkpoint"
)
DATA_URL = (
    f"https://huggingface.co/datasets/neashton/drivaerml/resolve/{DATA_REVISION}/run_1"
)
ASSETS = {
    "checkpoint": {
        "filename": "Transolver.0.501.mdlus",
        "url": f"{MODEL_URL}/Transolver.0.501.mdlus",
        "size": 39188235,
        "sha256": "eb98f399a050a8f8a24919335c61642e4a835bd4044f7e21abec231aa31fd82c",
    },
    "stats": {
        "filename": "global_stats.json",
        "url": f"{MODEL_URL}/global_stats.json",
        "size": 1375,
        "sha256": "c4b19eaca6158219570232ce094e6ef48b8705eb727b335441a310c7479b6b25",
    },
    "vtp": {
        "filename": "boundary_1.vtp",
        "url": f"{DATA_URL}/boundary_1.vtp",
        "size": 659606189,
        "sha256": "01d388402dad7a783db9c666ddb18e6db745aac16a3193c275e0726dd108bb40",
    },
    "stl": {
        "filename": "drivaer_1.stl",
        "url": f"{DATA_URL}/drivaer_1.stl",
        "size": 142385186,
        "sha256": "411e6651284a26fc94924106b833fd79febc6deba63922c929dd8acfc99720d2",
    },
}
CHUNK_SIZE = 1024 * 1024


def _verified(path, descriptor):
    if not path.is_file() or path.stat().st_size != descriptor["size"]:
        return False
    return sha256(path) == descriptor["sha256"]


def _download_once(destination, descriptor):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(
                descriptor["url"], headers={"User-Agent": "physicsnemo-native-qa"}
            )
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=60) as response:
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    size += len(chunk)
                    if size > descriptor["size"]:
                        raise ValueError(
                            f"{destination.name}: download exceeds pinned size"
                        )
                    output.write(chunk)
                    digest.update(chunk)
            if size != descriptor["size"]:
                raise ValueError(
                    f"{destination.name}: downloaded size does not match pin"
                )
            if digest.hexdigest() != descriptor["sha256"]:
                raise ValueError(
                    f"{destination.name}: downloaded SHA256 does not match pin"
                )
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download(destination, descriptor):
    for attempt in range(1, 4):
        print(
            f"Transolver assets: downloading {destination.name} (attempt {attempt}/3)",
            flush=True,
        )
        try:
            _download_once(destination, descriptor)
            print(f"Transolver assets: verified {destination.name}", flush=True)
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
        ) as error:
            if isinstance(error, urllib.error.HTTPError):
                retryable = error.code in (408, 429, 500, 502, 503, 504)
                error.close()
                if not retryable:
                    raise
            if attempt == 3:
                raise
            time.sleep(attempt)


def prepare_assets(cache_root: Path) -> Path:
    """Return a verified manifest; warm caches work without network access."""
    identity = hashlib.sha256(json.dumps(ASSETS, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    directory = Path(cache_root).resolve() / identity
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    # Serialize publication/repair across jobs sharing the same mounted cache.
    with (directory / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid = {
            name: _verified(directory / descriptor["filename"], descriptor)
            for name, descriptor in ASSETS.items()
        }
        if not all(valid.values()):
            manifest.unlink(missing_ok=True)
        for name, descriptor in ASSETS.items():
            destination = directory / descriptor["filename"]
            if valid[name]:
                print(
                    f"Transolver assets: reusing verified {destination.name}",
                    flush=True,
                )
            else:
                _download(destination, descriptor)
        document = {
            "format_version": 1,
            "revision": f"model:{MODEL_REVISION};data:{DATA_REVISION}",
            **{
                name: {
                    "path": item["filename"],
                    "sha256": item["sha256"],
                    "source_url": item["url"],
                }
                for name, item in ASSETS.items()
            },
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=directory,
                prefix=".manifest.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                json.dump(document, output, indent=2)
                output.write("\n")
            os.replace(temporary, manifest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Transolver assets: manifest ready at {manifest}", flush=True)
    return manifest
