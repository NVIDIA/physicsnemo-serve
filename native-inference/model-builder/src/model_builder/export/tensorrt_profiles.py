"""Explicit TensorRT precision profiles and their verified native dependencies."""

from collections.abc import Mapping
import ctypes
import hashlib
from pathlib import Path


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


def validate_tensorrt_profile(name: str) -> str:
    if not isinstance(name, str) or name not in (
        "baseline",
        "layout-order-exact",
        "layout-order-exact-v2",
        "geotransolver-exact",
        "geotransolver-exact-v2",
        "domino-surface-exact",
    ):
        raise ValueError(
            "tensorrt_profile must be baseline, layout-order-exact, layout-order-exact-v2, "
            "geotransolver-exact, geotransolver-exact-v2 or domino-surface-exact"
        )
    return name


def plugin_names(profile):
    validate_tensorrt_profile(profile)
    if profile == "baseline":
        return ()
    if profile == "layout-order-exact-v2":
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
        if trt.get_plugin_registry().get_creator(creator, "1", "") is None:
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
    if profile == "layout-order-exact-v2":
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
    return counts


def required_operators(profile="layout-order-exact"):
    return [
        {"id": "pnmir.tensorrt-" + name.replace("_", "-"), "abi": "1"}
        for name in plugin_names(profile)
    ]
