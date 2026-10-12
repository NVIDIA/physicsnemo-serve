"""Validate the requested GPU architecture in the selected builder environment."""

import re


def validate_target(device: str, required_gpu_arch: str | None) -> None:
    """Check syntax without importing Torch or querying host GPU hardware."""
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if required_gpu_arch is None:
        return
    if not isinstance(required_gpu_arch, str) or not re.fullmatch(
        r"sm[0-9]{2,3}", required_gpu_arch
    ):
        raise ValueError(
            "required GPU architecture must be sm followed by two or three digits, such as sm90"
        )
    if device == "cpu":
        raise ValueError("a required GPU architecture requires a CUDA device")


def check_target(device: str, required_gpu_arch: str | None) -> dict | None:
    """Enforce exact compute capability before loading or executing a model."""
    validate_target(device, required_gpu_arch)
    if required_gpu_arch is None:
        return None

    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the selected builder environment")
    index = int(device.partition(":")[2] or 0)
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(
            f"requested device {device} but only {count} CUDA devices are visible"
        )
    major, minor = torch.cuda.get_device_capability(index)
    actual = f"sm{major}{minor}"
    if actual != required_gpu_arch:
        raise ValueError(
            f"required GPU architecture {required_gpu_arch}, but selected device {device} has {actual}"
        )
    return {
        "required_gpu_arch": required_gpu_arch,
        "actual_gpu_arch": actual,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(index),
    }
