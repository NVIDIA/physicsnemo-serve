#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generic Lepton.AI deployment script.
#
# Builds an image from either physicsnemo-serve (internal GitLab) or earth2studio
# (public GitHub) — or any custom repo — and creates or updates a Lepton
# endpoint with it.
#
# Source presets (selected via --source):
#
#   physicsnemo-serve:
#     URL:         (from deploy/config.yaml)
#     Make target: make image
#     Image:       ${DOCKER_REPO}/scicomp-ferroflux:${IMAGE_TAG}
#     Tag var:     IMAGE_TAG
#     Port:        8001
#     Doc:         physicsnemo-serve/docs/lepton-deployment.md
#
#   earth2studio:
#     URL:         https://github.com/NVIDIA/earth2studio
#     Make target: make container-service
#     Image:       ${DOCKER_REPO}/earth2studio-scicomp:${E2S_IMAGE_TAG}
#     Tag var:     E2S_IMAGE_TAG
#     Port:        8000
#     Doc:         earth2studio/serve/DEPLOY.md
#
#   custom:
#     All build/image/port fields must be supplied via flags.
#
# Cluster (node group), GPU resource shape, image registry, NFS mount, and
# all Lepton tokens are configurable. The repo can be supplied as a local
# checkout (--repo-path) or cloned on demand (--git-url [+ --git-ref]).
#
# Note: the exact `lep` CLI flag names depend on your leptonai version.
# Use --dry-run first to confirm the invocation looks right; tweak the
# create/update lines if your CLI uses different flag names.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/deploy/config.sh"

# ---------------------------------------------------------------------------
# Defaults (env vars override)

SOURCE="${SOURCE:-}"

REPO_PATH="${REPO_PATH:-}"
GIT_URL="${GIT_URL:-}"
GIT_REF="${GIT_REF:-main}"
GIT_TOKEN="${GIT_TOKEN:-${GITLAB_TOKEN:-}}"     # used for private GitLab clones

DOCKER_REPO="${DOCKER_REPO:-$(_cfg docker_registry)}"
IMAGE_NAME="${IMAGE_NAME:-}"                     # filled by --source preset if empty
IMAGE_TAG="${IMAGE_TAG:-}"                       # auto-generated if empty
MAKE_TARGET="${MAKE_TARGET:-}"                   # filled by --source preset if empty
MAKE_TAG_VAR="${MAKE_TAG_VAR:-}"                 # IMAGE_TAG (physicsnemo-serve) or E2S_IMAGE_TAG (earth2studio)

LEPTON_WORKSPACE_ID="${LEPTON_WORKSPACE_ID:-$(_cfg lepton_workspace_id)}"
LEPTON_WORKSPACE_URL="${LEPTON_WORKSPACE_URL:-}"
LEPTON_WORKSPACE_TOKEN="${LEPTON_WORKSPACE_TOKEN:-}"
LEPTON_NODE_GROUP="${LEPTON_NODE_GROUP:-$(_cfg lepton_node_group)}"
LEPTON_RESOURCE_SHAPE="${LEPTON_RESOURCE_SHAPE:-gpu.h100-sxm}"
LEPTON_LUSTRE_STORAGE="${LEPTON_LUSTRE_STORAGE:-lustre}"
LEPTON_ENDPOINT_NAME="${LEPTON_ENDPOINT_NAME:-}"
LEPTON_ENDPOINT_TOKEN="${LEPTON_ENDPOINT_TOKEN:-}"
LEPTON_PORT="${LEPTON_PORT:-}"                   # filled by --source preset if empty
LEPTON_PULL_SECRET="${LEPTON_PULL_SECRET:-$(_cfg pull_secret)}"
LEPTON_NFS_PATH="${LEPTON_NFS_PATH:-$(_cfg nfs_mount_base)/${USER}}"
LEPTON_MOUNT_TARGET="${LEPTON_MOUNT_TARGET:-/outputs}"
LEPTON_CONTAINER_ENVS=()

SKIP_BUILD=0
SKIP_PUSH=0
WAIT_FOR_READY=1
READY_TIMEOUT_SECONDS=600
DRY_RUN=0

CLONE_CACHE_DIR="${CLONE_CACHE_DIR:-${HOME}/.cache/lepton-deploy}"

# ---------------------------------------------------------------------------
# Usage

usage() {
    cat <<'EOF'
Usage: deploy-to-lepton.sh --source {physicsnemo-serve|earth2studio|custom} [options]

Source presets (use --source to select):
  --source physicsnemo-serve  Internal GitLab repo; make image; port 8001
  --source earth2studio     Public GitHub repo; make container-service; port 8000
  --source custom           Provide all build/image/port options explicitly

Required (flag or env var):
  --workspace-id ID         LEPTON_WORKSPACE_ID    Lepton workspace id (from deploy/config.yaml or env)
  --workspace-token TOKEN   LEPTON_WORKSPACE_TOKEN Workspace token (NGC API key for DGX Cloud
                                                   Lepton, or value from a browser login)
  --endpoint-token TOKEN    LEPTON_ENDPOINT_TOKEN  Bearer token for the endpoint

  The script combines workspace-id and workspace-token into the
  credentials string '<id>:<token>' that `lep login -c` requires.

Optional:
  --workspace-url URL       LEPTON_WORKSPACE_URL   Passed to `lep login -u` when set
                                                   (e.g. https://<workspace>.dgxc.lepton.run)

Source location (one of, unless --skip-build):
  --repo-path PATH          REPO_PATH              Local checkout to build from
  --git-url URL             GIT_URL                Clone from URL into cache (no auth for GitHub)
  --git-ref REF             GIT_REF                Branch/tag/sha (default: main)
  --git-token TOKEN         GIT_TOKEN/GITLAB_TOKEN PAT for private GitLab clones

Build / image:
  --image-name NAME         IMAGE_NAME             Full image name (no tag); overrides preset
  --image-tag TAG           IMAGE_TAG              Image tag (default: v0.1.<YYYYMMDD>.0)
  --registry REPO           DOCKER_REPO            Docker repo prefix (default: from deploy/config.yaml)
  --make-target TARGET      MAKE_TARGET            `make` target; overrides preset
  --make-tag-var NAME       MAKE_TAG_VAR           Makefile var that holds the tag

Cluster / placement:
  --node-group NG           LEPTON_NODE_GROUP      Cluster (default: from deploy/config.yaml)
  --resource-shape SHAPE    LEPTON_RESOURCE_SHAPE  GPU shape (default: gpu.h100-sxm; see `lep deployment create -h`)
  --endpoint-name NAME      LEPTON_ENDPOINT_NAME   Endpoint name (default: $USER-<source>-ep)
  --port PORT               LEPTON_PORT            Container port; overrides preset
  --pull-secret SECRET      LEPTON_PULL_SECRET     Image pull secret (default: from deploy/config.yaml)
  --env NAME=VALUE                                 Container environment variable

Mounts:
  --nfs-path PATH           LEPTON_NFS_PATH        Host NFS path (default: from deploy/config.yaml)
  --mount-target PATH       LEPTON_MOUNT_TARGET    Container mount target (default: /outputs)
  --lustre-storage NAME     LEPTON_LUSTRE_STORAGE  NFS storage volume name (default: lepton-shared-fs)

Flow:
  --skip-build                                     Skip build step
  --skip-push                                      Skip docker push
  --no-wait                                        Don't poll for Ready
  --dry-run                                        Print commands without executing
  -h, --help                                       Show this help

Examples:
  # Deploy physicsnemo-serve from an existing local checkout
  deploy-to-lepton.sh --source physicsnemo-serve \
      --repo-path ~/inference/physicsnemo-serve \
      --workspace-id <WORKSPACE_ID> --workspace-token "$LEPTON_LOGIN_TOKEN" \
      --endpoint-token "$EP_TOKEN"

  # Deploy earth2studio by cloning a tag
  deploy-to-lepton.sh --source earth2studio \
      --git-url https://github.com/NVIDIA/earth2studio --git-ref v0.15.0 \
      --workspace-id <WORKSPACE_ID> --workspace-token "$LEPTON_LOGIN_TOKEN" \
      --endpoint-token "$EP_TOKEN"
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --repo-path) REPO_PATH="$2"; shift 2 ;;
        --git-url) GIT_URL="$2"; shift 2 ;;
        --git-ref) GIT_REF="$2"; shift 2 ;;
        --git-token) GIT_TOKEN="$2"; shift 2 ;;
        --image-name) IMAGE_NAME="$2"; shift 2 ;;
        --image-tag) IMAGE_TAG="$2"; shift 2 ;;
        --registry) DOCKER_REPO="$2"; shift 2 ;;
        --make-target) MAKE_TARGET="$2"; shift 2 ;;
        --make-tag-var) MAKE_TAG_VAR="$2"; shift 2 ;;
        --workspace-id) LEPTON_WORKSPACE_ID="$2"; shift 2 ;;
        --workspace-url) LEPTON_WORKSPACE_URL="$2"; shift 2 ;;
        --workspace-token) LEPTON_WORKSPACE_TOKEN="$2"; shift 2 ;;
        --endpoint-token) LEPTON_ENDPOINT_TOKEN="$2"; shift 2 ;;
        --node-group) LEPTON_NODE_GROUP="$2"; shift 2 ;;
        --resource-shape) LEPTON_RESOURCE_SHAPE="$2"; shift 2 ;;
        --lustre-storage) LEPTON_LUSTRE_STORAGE="$2"; shift 2 ;;
        --endpoint-name) LEPTON_ENDPOINT_NAME="$2"; shift 2 ;;
        --port) LEPTON_PORT="$2"; shift 2 ;;
        --pull-secret) LEPTON_PULL_SECRET="$2"; shift 2 ;;
        --env) LEPTON_CONTAINER_ENVS+=("$2"); shift 2 ;;
        --nfs-path) LEPTON_NFS_PATH="$2"; shift 2 ;;
        --mount-target) LEPTON_MOUNT_TARGET="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-push) SKIP_PUSH=1; shift ;;
        --no-wait) WAIT_FOR_READY=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Apply --source preset

if [[ -z "$SOURCE" ]]; then
    echo "Error: --source is required (one of: physicsnemo-serve, earth2studio, custom)" >&2
    exit 2
fi

# Generate an 8-digit random suffix to avoid endpoint name collisions.
_SALT="$(shuf -i 10000000-99999999 -n1 2>/dev/null || printf '%08d' $((RANDOM * RANDOM % 100000000)))"

case "$SOURCE" in
    physicsnemo-serve)
        : "${MAKE_TARGET:=image}"
        : "${MAKE_TAG_VAR:=IMAGE_TAG}"
        : "${IMAGE_NAME:=${DOCKER_REPO}/$(_cfg image_name)}"
        : "${LEPTON_PORT:=8001}"
        : "${GIT_URL:=$(_cfg repo_url)}"
        : "${LEPTON_ENDPOINT_NAME:=${USER}-pnserve-ep-${_SALT}}"
        ;;
    earth2studio)
        : "${MAKE_TARGET:=container-service}"
        : "${MAKE_TAG_VAR:=E2S_IMAGE_TAG}"
        : "${IMAGE_NAME:=${DOCKER_REPO}/$(_cfg python_service_image)}"
        : "${LEPTON_PORT:=8000}"
        : "${GIT_URL:=https://github.com/NVIDIA/earth2studio}"
        : "${LEPTON_ENDPOINT_NAME:=${USER}-earth2studio-ep-${_SALT}}"
        ;;
    custom)
        : "${LEPTON_ENDPOINT_NAME:=${USER}-custom-ep-${_SALT}}"
        if [[ "$SKIP_BUILD" -eq 0 ]]; then
            for var_name in MAKE_TARGET MAKE_TAG_VAR IMAGE_NAME LEPTON_PORT; do
                if [[ -z "${!var_name}" ]]; then
                    echo "Error: --source custom requires $var_name (--make-target / --make-tag-var / --image-name / --port)" >&2
                    exit 2
                fi
            done
        fi
        ;;
    *)
        echo "Error: unknown --source '$SOURCE' (expected: physicsnemo-serve, earth2studio, custom)" >&2
        exit 2
        ;;
esac

# ---------------------------------------------------------------------------
# Validation

require() {
    local label="$1" value="$2"
    if [[ -z "$value" ]]; then
        echo "Error: $label is required" >&2
        echo "  pass the matching --flag or set the env var (see --help)" >&2
        exit 2
    fi
}

require "workspace id"     "$LEPTON_WORKSPACE_ID"
require "endpoint token"   "$LEPTON_ENDPOINT_TOKEN"

if ! command -v lep >/dev/null 2>&1; then
    echo "Error: 'lep' CLI not found. Install with: pip install -U leptonai" >&2
    exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 || "$SKIP_PUSH" -eq 0 ]]; then
    command -v docker >/dev/null 2>&1 || {
        echo "Error: docker not found (needed unless both --skip-build and --skip-push)" >&2
        exit 1
    }
fi

if [[ "$SKIP_BUILD" -eq 0 && -z "$REPO_PATH" && -z "$GIT_URL" ]]; then
    echo "Error: provide --repo-path or --git-url (or pass --skip-build)" >&2
    exit 2
fi

if [[ -z "$IMAGE_TAG" ]]; then
    IMAGE_TAG="v0.1.$(date +%Y%m%d).0"
fi

if [[ "$IMAGE_TAG" == *"/"* ]]; then
    IMAGE_FULL="$IMAGE_TAG"
else
    IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Echo where the source code is coming from, so it is visible before any
# build or clone runs. With --source physicsnemo-serve/earth2studio, GIT_URL is
# set by the preset if not given explicitly — surface that here.
if [[ -n "$REPO_PATH" ]]; then
    echo "==> Source repo: $REPO_PATH (local)"
elif [[ -n "$GIT_URL" ]]; then
    echo "==> Source repo: $GIT_URL (ref: $GIT_REF)"
fi

# ---------------------------------------------------------------------------
# Helpers

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

# Read field $1 from JSON on stdin (handles object or list).
json_field() {
    local field="$1"
    python3 - "$field" <<'PY'
import json, sys
field = sys.argv[1]
data = json.load(sys.stdin)
items = data if isinstance(data, list) else [data]
for item in items:
    if isinstance(item, dict) and field in item and item[field] is not None:
        print(item[field])
        break
PY
}

deployment_exists() {
    local name="$1"
    lep deployment list -j 2>/dev/null | python3 - "$name" <<'PY' || true
import json, sys
name = sys.argv[1]
data = json.load(sys.stdin)
items = data if isinstance(data, list) else data.get("items", [])
for item in items:
    md = item.get("metadata") if isinstance(item, dict) else None
    n = md.get("name") if isinstance(md, dict) else item.get("name") if isinstance(item, dict) else None
    if n == name:
        print("EXISTS")
        break
PY
}

deployment_field() {
    local name="$1" field="$2"
    lep deployment status -n "$name" -j 2>/dev/null | json_field "$field"
}

# ---------------------------------------------------------------------------
# Resolve repo path (clone if needed)

if [[ "$SKIP_BUILD" -eq 0 && -z "$REPO_PATH" && -n "$GIT_URL" ]]; then
    # Catch the common pitfall: GitLab clones need a PAT.
    if [[ -z "$GIT_TOKEN" && "$GIT_URL" == *gitlab-master.nvidia.com* ]]; then
        echo "Error: cloning from $GIT_URL requires a Personal Access Token." >&2
        echo "  Pass --git-token <PAT> or set GITLAB_TOKEN (scope: read_repository)." >&2
        echo "  Or supply a local checkout via --repo-path." >&2
        exit 2
    fi

    # Feed the PAT to git via GIT_ASKPASS so it never lands in argv, in this
    # script's stdout, or in $REPO_PATH/.git/config. The askpass shim only
    # echoes $GIT_TOKEN; the secret itself stays in the environment.
    if [[ -n "$GIT_TOKEN" && "$GIT_URL" == https://* ]]; then
        ASKPASS_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/lepton-git-askpass.XXXXXX")"
        trap 'rm -f "$ASKPASS_SCRIPT"' EXIT
        cat > "$ASKPASS_SCRIPT" <<'ASKPASS_EOF'
#!/usr/bin/env bash
echo "$GIT_TOKEN"
ASKPASS_EOF
        chmod 700 "$ASKPASS_SCRIPT"
        export GIT_TOKEN GIT_ASKPASS="$ASKPASS_SCRIPT" GIT_TERMINAL_PROMPT=0
        CLONE_URL="https://oauth2@${GIT_URL#https://}"
    else
        CLONE_URL="$GIT_URL"
    fi

    mkdir -p "$CLONE_CACHE_DIR"
    REPO_NAME="$(basename "${GIT_URL%.git}")"
    REPO_PATH="${CLONE_CACHE_DIR}/${REPO_NAME}"
    if [[ -d "$REPO_PATH/.git" ]]; then
        echo "==> Updating cached clone at $REPO_PATH"
        run git -C "$REPO_PATH" fetch --tags --prune origin
        run git -C "$REPO_PATH" checkout "$GIT_REF"
        run git -C "$REPO_PATH" pull --ff-only origin "$GIT_REF" || {
            echo "Warning: git pull failed; building from existing cache at $REPO_PATH" >&2
            echo "  Run 'git -C $REPO_PATH log -1 --oneline' to confirm the cached ref." >&2
        }
    else
        echo "==> Cloning $GIT_URL into $REPO_PATH (ref: $GIT_REF)"
        run git clone --depth 1 --branch "$GIT_REF" "$CLONE_URL" "$REPO_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Summary

cat <<EOF
==> Configuration
    source         : $SOURCE
    repo-path      : ${REPO_PATH:-<skipped build>}
    workspace      : $LEPTON_WORKSPACE_ID
    node-group     : $LEPTON_NODE_GROUP
    resource-shape : $LEPTON_RESOURCE_SHAPE
    endpoint-name  : $LEPTON_ENDPOINT_NAME
    image          : $IMAGE_FULL
    make-target    : ${MAKE_TARGET:-<n/a>}
    tag-var        : ${MAKE_TAG_VAR:-<n/a>}
    port           : $LEPTON_PORT
    mount          : $LEPTON_NFS_PATH -> $LEPTON_MOUNT_TARGET (storage: $LEPTON_LUSTRE_STORAGE)
    pull-secret    : $LEPTON_PULL_SECRET
    skip-build=$SKIP_BUILD skip-push=$SKIP_PUSH dry-run=$DRY_RUN
EOF

# ---------------------------------------------------------------------------
# Build

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "==> Building image (make $MAKE_TARGET)"
    [[ -d "$REPO_PATH" ]] || { echo "repo-path not found: $REPO_PATH" >&2; exit 1; }
    run make -C "$REPO_PATH" "$MAKE_TARGET" \
        "${MAKE_TAG_VAR}=${IMAGE_TAG}" \
        "DOCKER_REPO=${DOCKER_REPO}"
else
    echo "==> Skipping build"
fi

# ---------------------------------------------------------------------------
# Push

if [[ "$SKIP_PUSH" -eq 0 ]]; then
    echo "==> Pushing $IMAGE_FULL"
    run docker push "$IMAGE_FULL"
else
    echo "==> Skipping push"
fi

# ---------------------------------------------------------------------------
# Login

echo "==> Logging into Lepton workspace"
# lep v0.27+ takes -c <workspace-id>:<token>  (and optional -u <workspace-url>).
# Don't pipe through run(); the credentials string contains the workspace
# token, which run()'s `echo "+ $*"` would leak to stdout / CI logs.
if [[ -n "$LEPTON_WORKSPACE_TOKEN" ]]; then
    LEPTON_CREDENTIALS="${LEPTON_WORKSPACE_ID}:${LEPTON_WORKSPACE_TOKEN}"
    echo "+ lep login -c ${LEPTON_WORKSPACE_ID}:<redacted>${LEPTON_WORKSPACE_URL:+ -u $LEPTON_WORKSPACE_URL}"
elif [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
    echo "Error: --workspace-url requires --workspace-token so the script can log into that workspace" >&2
    exit 2
else
    echo "+ using existing lep login session"
fi
if [[ "$DRY_RUN" -eq 0 && -n "$LEPTON_WORKSPACE_TOKEN" ]]; then
    if [[ -n "$LEPTON_WORKSPACE_URL" ]]; then
        lep login -c "$LEPTON_CREDENTIALS" -u "$LEPTON_WORKSPACE_URL"
    else
        lep login -c "$LEPTON_CREDENTIALS"
    fi
fi

# ---------------------------------------------------------------------------
# Create or update endpoint

echo "==> Creating or replacing endpoint: $LEPTON_ENDPOINT_NAME"
# --rerun handles both 'new' and 'already exists' in one call: if an
# endpoint of the same name exists, it's shut down and recreated with
# the new spec. Acceptable for dev/test; for production prefer
# `lep deployment update` instead.
deploy_args=(
    --rerun
    --name "$LEPTON_ENDPOINT_NAME"
    --container-image "$IMAGE_FULL"
    --container-port "$LEPTON_PORT"
    --node-group "$LEPTON_NODE_GROUP"
    --resource-shape "$LEPTON_RESOURCE_SHAPE"
    --image-pull-secrets "$LEPTON_PULL_SECRET"
    --mount "${LEPTON_NFS_PATH}:${LEPTON_MOUNT_TARGET}:node-nfs:${LEPTON_LUSTRE_STORAGE}"
    --tokens "$LEPTON_ENDPOINT_TOKEN"
    --replicas-static 1
)
if ((${#LEPTON_CONTAINER_ENVS[@]} > 0)); then
    for env_pair in "${LEPTON_CONTAINER_ENVS[@]}"; do
        deploy_args+=(--env "$env_pair")
    done
fi
# Don't pipe through run(); --tokens carries the endpoint bearer token.
preview_args=()
for arg in "${deploy_args[@]}"; do
    if [[ "$arg" == "$LEPTON_ENDPOINT_TOKEN" ]]; then
        preview_args+=("<redacted>")
    else
        preview_args+=("$arg")
    fi
done
echo "+ lep deployment create ${preview_args[*]}"
if [[ "$DRY_RUN" -eq 0 ]]; then
    lep deployment create "${deploy_args[@]}"
fi

# ---------------------------------------------------------------------------
# Wait for ready

if [[ "$WAIT_FOR_READY" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    echo "==> Waiting for endpoint Ready (up to ${READY_TIMEOUT_SECONDS}s)"
    end_time=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
    last_summary=""
    while [[ $(date +%s) -lt $end_time ]]; do
        # lep v0.27 uses `lep endpoint status -n NAME`. No -j flag; parse the
        # human-readable output.
        status_out="$(lep endpoint status -n "$LEPTON_ENDPOINT_NAME" 2>&1 || true)"
        summary="$(printf '%s\n' "$status_out" | grep -iE '^[[:space:]]*(state|status|endpoint)' | head -3 | tr -d '\r' || true)"
        if [[ -n "$summary" && "$summary" != "$last_summary" ]]; then
            printf '%s\n' "$summary" | sed 's/^/    /'
            last_summary="$summary"
        fi
        if printf '%s\n' "$status_out" | grep -qiE '^[[:space:]]*State:.*(Ready|Running)'; then
            break
        fi
        sleep 15
    done
fi

# ---------------------------------------------------------------------------
# Final summary

if [[ "$DRY_RUN" -eq 0 ]]; then
    # Show the full endpoint status so the user gets the URL and any errors.
    echo "==> Final endpoint status:"
    lep endpoint status -n "$LEPTON_ENDPOINT_NAME" 2>&1 | sed 's/^/    /' || true
    URL="$(lep endpoint status -n "$LEPTON_ENDPOINT_NAME" 2>&1 | grep -iE '^[[:space:]]*Endpoint[[:space:]]*[:=]' | head -1 | sed -E 's/^[[:space:]]*Endpoint[[:space:]]*[:=][[:space:]]*//' | tr -d '\r' || true)"
    # lep v0.44+ doesn't print an "Endpoint:" URL line; construct it from
    # the known pattern: https://{workspace-id}-{endpoint-name}.xenon.lepton.run
    if [[ -z "$URL" ]]; then
        URL="https://${LEPTON_WORKSPACE_ID}-${LEPTON_ENDPOINT_NAME}.xenon.lepton.run"
    fi
else
    URL="(dry-run; not deployed)"
fi

cat <<EOF

==> Deployment summary
    Source:   $SOURCE
    Endpoint: $LEPTON_ENDPOINT_NAME
    Image:    $IMAGE_FULL
    Cluster:  $LEPTON_NODE_GROUP
    Shape:    $LEPTON_RESOURCE_SHAPE
    Port:     $LEPTON_PORT
    URL:      $URL

To send traffic:
    export SERVICE_URL=$URL
    export EP_TOKEN=<your endpoint token>
    curl -X POST "\${SERVICE_URL}/v1/infer/<workflow>/run" \\
         -H "Authorization: Bearer \${EP_TOKEN}" \\
         -H 'Content-Type: application/json' \\
         -d '<json>'
EOF
