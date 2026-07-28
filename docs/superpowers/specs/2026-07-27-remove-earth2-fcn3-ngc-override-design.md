# Remove the Earth2 FCN3 NGC Override

## Objective

Remove the explicit public NVIDIA NGC package override from the generic
`earth2-deterministic-batch` plugin. FCN3 must return to Earth2Studio's default
package-loading contract, matching DLWP and FCN. Rebuild the runtime image and
redeploy the existing Lepton endpoint without changing its infrastructure
configuration.

## Scope

The rollback removes:

- `ModelSpec.pretrained_uri`
- FCN3's pinned
  `ngc://models/nvidia/earth-2/fourcastnet3@0.1.0` registry value
- the conditional `from_pretrained()` path in `_load_model()`
- NGC-specific fixture behavior and assertions
- documentation that recommends or describes the public NGC package
- the patch-only plugin version `1.1.1`

The rollback preserves:

- DLWP, FCN, and FCN3 support
- the immutable model registry
- model-specific resource and batching profiles
- normalized model names and homogeneous batch keys
- lazy model loading and per-model inter-process cache locks
- process-local runtime reuse
- model-switch eviction and cleanup
- per-batch load-failure containment and retry behavior
- the existing request and result schemas

## Runtime Design

All registry entries use the same model-loading flow:

1. Resolve the Earth2Studio model class from the selected `ModelSpec`.
2. Enter the selected model's cache lock.
3. Call `model_class.load_default_package()`.
4. Call `model_class.load_model(package)`.
5. Move the loaded model to the selected execution device.
6. Cache the completed `DeterministicBatchRuntime` only after both model and
   GFS datasource initialization succeed.

For FCN3, the current Earth2Studio checkout therefore controls the default
package URI and revision. The plugin does not select, replace, or fall back to
another registry.

## Contract and Version

The model enum remains:

```text
dlwp, fcn, fcn3
```

No request or result field changes. The plugin version returns from `1.1.1` to
`1.1.0` because `1.1.1` identified only the NGC package-source patch. The
replacement image receives a new immutable NVCR tag even though the plugin
contract version returns to `1.1.0`.

## Tests and Documentation

Focused tests must establish that:

- all three model classes call `load_default_package()` and
  `load_model(package)`
- FCN3 no longer calls `from_pretrained()`
- model locking, reuse, switching, cleanup, failure containment, ordering, and
  output registration remain unchanged
- the manifest reports version `1.1.0` and all three model names

The README and batching guide will state that every supported model uses its
Earth2Studio default package. All NGC URI and anonymous Hugging Face quota
workaround text will be removed.

## Image and Deployment

Build a new linux/amd64 image from the exact local source state and the existing
pinned Earth2Studio commit. Run in-image import, package-loader, and plugin
validation before pushing. Push the image to the existing NVCR repository and
verify its remote digest.

Update only the container image on the existing Lepton endpoint. Preserve the
endpoint's GPU shape, replica count, node group, mount, pull secret, port,
timeout, and authentication settings.

## Acceptance Criteria

- focused Python tests pass
- Ruff, shell syntax, and `git diff --check` pass
- the image is present in NVCR with a verified digest
- the Lepton endpoint reaches Ready with one ready replica
- five consecutive authenticated `/healthz` requests return HTTP 200
- live schema reports plugin version `1.1.0` and models `dlwp`, `fcn`, `fcn3`
- a DLWP request succeeds and registers a distinct `forecast_dataset`
- logs or a controlled loader probe confirm FCN3 follows
  `load_default_package()` rather than the removed NGC `from_pretrained()` path

A live cold FCN3 forecast is attempted only as a bounded validation. Anonymous
Hugging Face throttling is an external dependency failure and does not trigger
reintroduction of the NGC override.

## Rollback

If the replacement image fails readiness or regresses DLWP execution, restore
the currently Ready version-10 image:

```text
nvcr.io/dycvht5ows21/scicomp-ferroflux:batch-multimodel-20260727T214300Z
```

Do not restore the NGC source override in code without a new design decision.
