"""Explicit AOTInductor options supported by Model Builder."""

from .aoti_profiles import validate_aoti_profile

SUPPORTED_AOTI_OPTIONS = frozenset(
    {
        "max_autotune",
        "epilogue_fusion",
        "shape_padding",
        "coordinate_descent_tuning",
    }
)


def validate_aoti_options(options, profile="baseline"):
    """Validate explicit overrides without importing any framework."""
    validate_aoti_profile(profile)
    if not isinstance(options, dict):
        raise ValueError("aoti_options must be an object")
    unknown = options.keys() - SUPPORTED_AOTI_OPTIONS
    if unknown:
        raise ValueError(
            "Unsupported AOTI option: " + ", ".join(sorted(map(str, unknown)))
        )
    for key, value in options.items():
        if type(value) is not bool:
            raise ValueError(f"aoti_options.{key} must be a boolean")
    if options.get("epilogue_fusion") and not options.get("max_autotune"):
        raise ValueError("aoti_options.epilogue_fusion=true requires max_autotune=true")
    if profile in ("aten-boundary-exact-v2", "aten-boundary-exact-v3") and any(
        options.values()
    ):
        raise ValueError(
            f"Enabled performance options are not qualified with {profile}; "
            "use aoti_profile=baseline to experiment, or disable these options"
        )
    return dict(options)
