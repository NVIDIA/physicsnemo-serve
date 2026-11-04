"""Explicit TensorRT precision profiles and their verified native dependencies."""

from collections.abc import Mapping
from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
import re


EXACT_PLUGIN_NAMES = (
    "exact_linear",
    "exact_gemm",
    "exact_token_sum",
    "exact_slice_bmm",
    "exact_layer_norm",
    "exact_softmax",
    "exact_attention",
    "exact_gelu",
)
_PLUGIN_CREATORS = (
    "PNMIRExactLinear",
    "PNMIRExactGemm",
    "PNMIRExactTokenSum",
    "PNMIRExactSliceBmm",
    "PNMIRExactLayerNorm",
    "PNMIRExactSoftmax",
    "PNMIRExactAttention",
    "PNMIRExactGelu",
)
GEOTRANSOLVER_PLUGIN_NAMES = (*EXACT_PLUGIN_NAMES, "exact_weighted_blend")
GEOTRANSOLVER_V2_PLUGIN_NAMES = (*GEOTRANSOLVER_PLUGIN_NAMES, "exact_deslice_bmm")
TRANSOLVER_V2_PLUGIN_NAMES = (*EXACT_PLUGIN_NAMES, "exact_deslice_bmm")
DOMINO_PLUGIN_NAMES = (
    "exact_linear",
    "exact_gelu",
    "exact_scalar_div",
    "exact_inverse_distance_blend",
)
# The prepared surface core requires exact linears in every learned stage.
DOMINO_LINEAR_PREFIXES = ("",)
_CREATORS_BY_NAME = dict(zip(EXACT_PLUGIN_NAMES, _PLUGIN_CREATORS, strict=True))
_CREATORS_BY_NAME["exact_weighted_blend"] = "PNMIRExactWeightedBlend"
_CREATORS_BY_NAME["exact_scalar_div"] = "PNMIRExactScalarDiv"
_CREATORS_BY_NAME["exact_inverse_distance_blend"] = "PNMIRExactInverseDistanceBlend"
_CREATORS_BY_NAME["exact_deslice_bmm"] = "PNMIRExactDesliceBmm"
PROVIDER_DEFAULT_EXACT = "provider-default-exact-v1"
PROVIDER_BLAS_NAMES = frozenset(
    ("exact_linear", "exact_gemm", "exact_slice_bmm", "exact_deslice_bmm")
)
PROVIDER_MATH_POLICY = {
    "name": "provider-default-v1",
    "compute_capability": "9.0",
    "float32_matmul_precision": "high",
    "allow_tf32_matmul": True,
    "allow_tf32_cudnn": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": False,
    "deterministic_algorithms": False,
    "preferred_blas_library": "cublas",
    "cublas_workspace_bytes": 32 * 1024 * 1024,
    "cublaslt_workspace_bytes": 1024 * 1024,
    "native_tensorrt_tf32": False,
}


@contextmanager
def provider_default_math(device=None):
    """Pinned H100 provider policy; restore every changed caller setting."""
    import torch

    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise ValueError("provider-default-exact-v1 requires the qualified sm90 stack")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is not None:
        pairs = re.findall(r":([0-9]+):([0-9]+)", workspace)
        if (
            not pairs
            or sum(int(size) * int(count) * 1024 for size, count in pairs) != 33554432
        ):
            raise ValueError("provider profile requires a 32 MiB cuBLAS workspace")
    if os.environ.get("CUBLASLT_WORKSPACE_SIZE", "1024") != "1024":
        raise ValueError("provider profile requires a 1 MiB cuBLASLt workspace")
    if os.environ.get("TORCH_CUBLASLT_UNIFIED_WORKSPACE", "0") == "1":
        raise ValueError("provider profile requires separate cuBLASLt workspace")
    precision = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn
    previous = (
        cudnn.allow_tf32,
        cudnn.benchmark,
        cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.preferred_blas_library(),
    )
    try:
        torch.set_float32_matmul_precision("high")
        cudnn.allow_tf32 = True
        cudnn.benchmark = False
        cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.preferred_blas_library("cublas")
        yield
    finally:
        cudnn.allow_tf32, cudnn.benchmark, cudnn.deterministic = previous[:3]
        torch.use_deterministic_algorithms(previous[3], warn_only=previous[4])
        torch.backends.cuda.preferred_blas_library(previous[5])
        torch.set_float32_matmul_precision(precision)


def validate_tensorrt_profile(name: str) -> str:
    if not isinstance(name, str) or name not in (
        "baseline",
        "layout-order-exact",
        "layout-order-exact-v2",
        "geotransolver-exact",
        "geotransolver-exact-v2",
        "domino-surface-exact",
        PROVIDER_DEFAULT_EXACT,
    ):
        raise ValueError(
            "tensorrt_profile must be baseline, layout-order-exact, layout-order-exact-v2, "
            "geotransolver-exact, geotransolver-exact-v2, domino-surface-exact "
            "or provider-default-exact-v1"
        )
    return name


def validate_profile_backends(profile, backends):
    validate_tensorrt_profile(profile)
    if profile == PROVIDER_DEFAULT_EXACT and list(backends) != ["tensorrt"]:
        raise ValueError("provider-default-exact-v1 requires a TensorRT-only build")


def plugin_names(profile):
    validate_tensorrt_profile(profile)
    if profile == "baseline":
        return ()
    if profile in ("layout-order-exact-v2", PROVIDER_DEFAULT_EXACT):
        return TRANSOLVER_V2_PLUGIN_NAMES
    if profile == "geotransolver-exact":
        return GEOTRANSOLVER_PLUGIN_NAMES
    if profile == "geotransolver-exact-v2":
        return GEOTRANSOLVER_V2_PLUGIN_NAMES
    if profile == "domino-surface-exact":
        return DOMINO_PLUGIN_NAMES
    return EXACT_PLUGIN_NAMES


def profile_version(profile):
    validate_tensorrt_profile(profile)
    if profile == "geotransolver-exact-v2":
        return 3
    return 2 if profile in ("layout-order-exact-v2", "geotransolver-exact") else 1


def selection_metadata(profile):
    validate_tensorrt_profile(profile)
    if profile == PROVIDER_DEFAULT_EXACT:
        return {"math_policy": dict(PROVIDER_MATH_POLICY)}
    if profile == "domino-surface-exact":
        return {
            "exact_linear_bias_name_prefixes": list(DOMINO_LINEAR_PREFIXES),
            "exact_gelu_after_exact_linear": True,
        }
    return {}


def requires_byte_identical(profile):
    validate_tensorrt_profile(profile)
    return profile in (
        "layout-order-exact-v2",
        "geotransolver-exact",
        "geotransolver-exact-v2",
        "domino-surface-exact",
        PROVIDER_DEFAULT_EXACT,
    )


def resolve_plugin_libraries(profile, libraries=None):
    names = plugin_names(profile)
    if profile == "baseline":
        if libraries:
            raise ValueError(
                "baseline TensorRT profile does not accept plugin libraries"
            )
        return {}
    if not isinstance(libraries, Mapping) or set(libraries) != set(names):
        count = {4: "four", 8: "eight", 9: "nine", 10: "ten"}[len(names)]
        raise ValueError(
            f"{profile} requires all {count} plugin libraries: " + ", ".join(names)
        )
    result = {}
    for name in names:
        path = Path(libraries[name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"TensorRT exact plugin library does not exist: {path}"
            )
        result[name] = path
    return result


def load_exact_plugins(trt, libraries, profile="layout-order-exact"):
    """Keep returned handles alive until serialization has finished."""
    handles = []
    records = {}
    for name in plugin_names(profile):
        creator = _CREATORS_BY_NAME[name]
        path = libraries[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
        version = (
            "2"
            if profile == PROVIDER_DEFAULT_EXACT and name in PROVIDER_BLAS_NAMES
            else "1"
        )
        if trt.get_plugin_registry().get_creator(creator, version, "") is None:
            raise RuntimeError(
                f"TensorRT exact plugin creator is not registered: {creator}"
            )
        records[name] = {"filename": path.name, "sha256": digest}
    return handles, records


def prepare_exact_graph(onnx, model, profile="layout-order-exact"):
    from functools import partial
    from . import tensorrt_exact_graphs as graphs

    validate_tensorrt_profile(profile)
    if profile == "baseline":
        return {}
    # Attention must be recognized before replacing its internal softmax.
    # Slice BMM must see the original transposes and token-sum output layout.
    transforms = (
        ("exact_linear", graphs._replace_linear_subgraphs),
        ("exact_gemm", graphs._replace_constant_rhs_matmuls),
        ("exact_layer_norm", graphs._replace_layer_norms),
        ("exact_attention", graphs._replace_attention_subgraphs),
        ("exact_token_sum", graphs._replace_token_sums),
        ("exact_slice_bmm", graphs._replace_slice_bmms),
        ("exact_gelu", graphs._replace_gelu_subgraphs),
        ("exact_softmax", graphs._replace_softmaxes),
    )
    if profile in ("layout-order-exact-v2", PROVIDER_DEFAULT_EXACT):
        # Deslicing needs the exact attention creator's physical BSHD output.
        transforms = (
            *transforms[:4],
            ("exact_deslice_bmm", graphs._replace_deslice_bmms),
            *transforms[4:],
        )
    elif profile in ("geotransolver-exact", "geotransolver-exact-v2"):
        transforms += (("exact_weighted_blend", graphs._replace_weighted_blends),)
        if profile == "geotransolver-exact-v2":
            # Prove both attention layouts only after recognizing their blend.
            transforms += (("exact_deslice_bmm", graphs._replace_deslice_bmms),)
    elif profile == "domino-surface-exact":
        transforms = (
            (
                "exact_linear",
                partial(
                    graphs._replace_linear_subgraphs,
                    bias_name_prefixes=DOMINO_LINEAR_PREFIXES,
                ),
            ),
            (
                "exact_gelu",
                partial(
                    graphs._replace_gelu_subgraphs,
                    exact_linear_sources_only=True,
                ),
            ),
            ("exact_scalar_div", graphs._replace_scalar_divs),
            ("exact_inverse_distance_blend", graphs._replace_inverse_distance_blends),
        )
    counts = {}
    for name, transform in transforms:
        counts[name] = transform(onnx, model)
        if counts[name] == 0:
            raise ValueError(f"{profile} found no supported {name} graph pattern")
    if profile == PROVIDER_DEFAULT_EXACT:
        blas_creators = {_CREATORS_BY_NAME[name] for name in PROVIDER_BLAS_NAMES}
        for node in model.graph.node:
            if node.op_type in ("MatMul", "Gemm"):
                raise ValueError(
                    "provider profile requires exact replacement of every matrix product"
                )
            if node.op_type in blas_creators:
                retained = [
                    field for field in node.attribute if field.name != "plugin_version"
                ]
                del node.attribute[:]
                node.attribute.extend(retained)
                for key, value in (
                    ("plugin_version", "2"),
                    ("compute_mode", 1),
                    ("cublas_workspace_bytes", 33554432),
                    ("cublaslt_workspace_bytes", 1048576),
                ):
                    node.attribute.append(onnx.helper.make_attribute(key, value))
    return counts


def required_operators(profile="layout-order-exact"):
    return [
        {
            "id": "pnmir.tensorrt-"
            + (
                "provider-default-"
                if profile == PROVIDER_DEFAULT_EXACT and name in PROVIDER_BLAS_NAMES
                else ""
            )
            + name.replace("_", "-"),
            "abi": "1",
        }
        for name in plugin_names(profile)
    ]
