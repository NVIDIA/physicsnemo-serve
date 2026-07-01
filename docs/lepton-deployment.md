# Lepton Deployment Guide

> **Note:** This guide references deployment values from `deploy/config.yaml`.
> Copy `deploy/config.example.yaml` to `deploy/config.yaml` and fill in your
> environment-specific values before following these instructions.

This document describes two ways to deploy a physicsnemo-serve image to a
Lepton.AI endpoint:

1. **Scripted (recommended)** — `scripts/deploy-to-lepton.sh` automates the
   full build → push → endpoint create/update flow with a single command.
2. **Manual fallback (dashboard)** — click-through via the Lepton dashboard,
   covered briefly at the bottom of this document.

Cluster (node group), GPU shape, image registry, NFS volume, and all auth
tokens are configurable via flags or environment variables on the script.

---

## 1. Scripted deployment

### Source presets

scripts/deploy-to-lepton.sh` can deploy **physicsnemo-serve** or **earth2studio**, from
  either a local checkout (`--repo-path`) or by cloning a git URL on demand
  (`--git-url` + `--git-token` for private GitLab). `--source` selects which
  preset (make target, image name, port, default URL) to apply:

| `--source` | Make target | Image name | Port | Default URL |
|---|---|---|---|---|
| `physicsnemo-serve`    | `make image`             | `${REGISTRY}/scicomp-ferroflux`   | `8001` | GitLab (auth needed) |
| `earth2studio` | `make container-service` | `${REGISTRY}/earth2studio-scicomp`| `8000` | GitHub (public) |
| `custom`       | required via flags       | required via flags                | required | — |

### Examples

```bash
# physicsnemo-serve cloned from GitLab (requires PAT with read_repository scope)
./scripts/deploy-to-lepton.sh --source physicsnemo-serve \
    --git-url   <REPO_URL> \
    --git-ref   main \
    --git-token "$GITLAB_TOKEN" \
    --workspace-id    <WORKSPACE_ID> \
    --workspace-token "$LEPTON_WORKSPACE_TOKEN" \
    --endpoint-token  "$EP_TOKEN" \
    --lustre-storage  lepton-shared-fs

# earth2studio cloned on demand from a tag (public GitHub — no token needed)
./scripts/deploy-to-lepton.sh --source earth2studio \
    --git-url https://github.com/NVIDIA/earth2studio --git-ref v0.15.0 \
    --workspace-id    <WORKSPACE_ID> \
    --workspace-token "$LEPTON_WORKSPACE_TOKEN" \
    --endpoint-token  "$EP_TOKEN" \
    --lustre-storage  lepton-shared-fs

# physicsnemo-serve from a local checkout instead of cloning
./scripts/deploy-to-lepton.sh --source physicsnemo-serve \
    --repo-path . \
    --workspace-id    <WORKSPACE_ID> \
    --workspace-token "$LEPTON_WORKSPACE_TOKEN" \
    --endpoint-token  "$EP_TOKEN" \
    --lustre-storage  lepton-shared-fs

# Preview without executing
./scripts/deploy-to-lepton.sh --source physicsnemo-serve \
    --workspace-id <WORKSPACE_ID> --workspace-token X --endpoint-token Y \
    --lustre-storage lepton-shared-fs --dry-run
```

Each `--source` preset has a default `--git-url` baked in, so you can omit
`--git-url` and the script will use the preset URL. The script prints
`==> Source repo: ...` at the start of the run so you can see which URL it
will use.

> **`lep login` credentials format.** `lep login -c` expects a single
> string `<workspace-id>:<token>`. The script combines `--workspace-id`
> and `--workspace-token` into that format internally — pass them as two
> separate flags, not pre-joined.
`scripts/deploy-to-lepton.sh` can deploy **physicsnemo-serve** or **earth2studio**, from
### Configurable knobs

| Flag | Env var | Default |
|---|---|---|
| `--workspace-id` | `LEPTON_WORKSPACE_ID` | **required** — Lepton workspace id (e.g. `<WORKSPACE_ID>`) |
| `--workspace-token` | `LEPTON_WORKSPACE_TOKEN` | **required** — workspace token (NGC API key for DGX Cloud Lepton) |
| `--endpoint-token` | `LEPTON_ENDPOINT_TOKEN` | **required** — Bearer auth for the endpoint |
| `--workspace-url` | `LEPTON_WORKSPACE_URL` | optional — passed to `lep login -u` when set |
| `--node-group` | `LEPTON_NODE_GROUP` | `<NODE_GROUP>` |
| `--resource-shape` | `LEPTON_RESOURCE_SHAPE` | `gpu.h100-sxm` (see `lep deployment create -h` for valid shapes) |
| `--endpoint-name` | `LEPTON_ENDPOINT_NAME` | `${USER}-<source>-ep` |
| `--port` | `LEPTON_PORT` | preset (`8001` physicsnemo-serve, `8000` earth2studio) |
| `--pull-secret` | `LEPTON_PULL_SECRET` | `<PULL_SECRET>` |
| `--nfs-path` | `LEPTON_NFS_PATH` | `<NFS_MOUNT_BASE>/${USER}` |
| `--mount-target` | `LEPTON_MOUNT_TARGET` | `/outputs` |
| `--lustre-storage` | `LEPTON_LUSTRE_STORAGE` | `lepton-shared-fs` — workspace storage volume name; combined into `<nfs-path>:<mount-target>:node-nfs:<storage>` for `lep deployment create --mount` |
| `--registry` | `DOCKER_REPO` | `<DOCKER_REGISTRY>` |
| `--image-tag` | `IMAGE_TAG` | `v0.1.<YYYYMMDD>.0` |
| `--git-url` / `--git-ref` / `--git-token` | `GIT_URL` / `GIT_REF` / `GITLAB_TOKEN` | preset URL, `main`, none |
| `--repo-path` | `REPO_PATH` | use a local checkout instead of cloning |
| `--skip-build` / `--skip-push` / `--no-wait` / `--dry-run` | — | flow control |

### Flow

1. **Resolve repo**: use `--repo-path` if given, otherwise clone `--git-url`
   (with `--git-token` for private GitLab) into
   `${CLONE_CACHE_DIR:-~/.cache/lepton-deploy}`.
2. `make <target> <TAG_VAR>=<tag> DOCKER_REPO=<registry>`.
3. `docker push <image>:<tag>`.
4. `lep login -c <workspace-id>:<token>` (with `-u <workspace-url>` if set).
5. `lep deployment create --rerun ...` — idempotent: creates the named
   endpoint, or shuts down and replaces an existing one with the new
   spec, in one call.
6. Poll `lep endpoint status -n <name>` until state is `Ready` or
   `Running`, then print the endpoint URL.

### Finding the right `--lustre-storage` value

`lep deployment create --mount` requires the workspace's NFS volume
name as the `MOUNT_FROM` piece. If the deploy fails with:

```
400 Error: invalid deployment spec: not all of the dedicated node groups
have local node shared volume <name>
```

…the volume name doesn't exist in that node group. Find the right one via:

- **CLI:** `lep storage ls-file-system` (some workspaces return 404 — in
  that case use the dashboard).
- **Dashboard:** <LEPTON_DASHBOARD_URL> → workspace →
  Storage tab. Note the volume name and pass it via
  `--lustre-storage <name>` or `export LEPTON_LUSTRE_STORAGE=<name>`.

Common values: `lepton-shared-fs` (default), `amlfs`.

### After deployment

```bash
export SERVICE_URL=<URL printed by the script>
export EP_TOKEN=<your endpoint token>

# List available workflows
curl -s -H "Authorization: Bearer ${EP_TOKEN}" \
    "${SERVICE_URL}/v1/infer/workflows" | jq

# Get a workflow's input schema
curl -s -H "Authorization: Bearer ${EP_TOKEN}" \
    "${SERVICE_URL}/v1/infer/<workflow>/schema" | jq

# Trigger a run
curl -X POST "${SERVICE_URL}/v1/infer/<workflow>/run" \
    -H "Authorization: Bearer ${EP_TOKEN}" \
    -H 'Content-Type: application/json' \
    -d @plugins/<workflow>/examples/default_request.json

# Poll status
curl -s -H "Authorization: Bearer ${EP_TOKEN}" \
    "${SERVICE_URL}/v1/infer/<workflow>/<run_id>/status" | jq
```

Interactive API docs are available at `${SERVICE_URL}/doc` (Swagger UI).

### Requirements

- `bash`, `python3`, `git`, `make`, `docker`
- [`leptonai`](https://pypi.org/project/leptonai/)
  (`pip install -U leptonai` in a Python ≥ 3.10 venv — provides `lep`)
- For private GitLab clones: a Personal Access Token with `read_repository`.
- NGC login for `nvcr.io` image pulls:
  ```
  echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  ```

The exact `lep` CLI flag names depend on your `leptonai` version. The
script targets v0.27 conventions (`--node-group`, `--tokens`,
`--replicas-static`, `--mount ...:node-nfs:<storage>`,
`lep endpoint status` for status). If your CLI uses different flag names,
adjust the `lep deployment create` invocation in
`scripts/deploy-to-lepton.sh`.

---

## 2. Manual fallback (dashboard)

For one-off deployments or when the script isn't an option:

1. **Build & push image** in your physicsnemo-serve checkout:
   ```bash
   make image            # builds <DOCKER_REGISTRY>/scicomp-ferroflux:<tag>
   docker push <DOCKER_REGISTRY>/scicomp-ferroflux:<tag>
   ```
2. **Create the endpoint** at
   <LEPTON_DASHBOARD_URL> → workspace → **Endpoints** →
   **Create Endpoint** → **Create from container image**.
   - **Name**: e.g. `${USER}-pnserve-ep`
   - **Resource**: node group `<NODE_GROUP>`, GPU shape
     `gpu.h100-sxm`.
   - **Image**: paste the full image:tag pushed above. **Server Port**
     `8001`. Leave the run command blank.
   - **Private Image Registry**: select `<PULL_SECRET>`.
   - **Storage**: `<NFS_MOUNT_BASE>/<your-path>` → `/outputs`,
     volume `lepton-shared-fs` (or whichever your workspace has).
   - **Access Tokens**: add an endpoint token; reuse across your
     endpoints if convenient.
3. **Click Create**, wait for **Ready**, copy the endpoint URL.

To change the image later, **Edit** the endpoint — it restarts with the
new image. The endpoint URL stays the same.

### Observability

To watch GPU utilization, memory, and request metrics for a running
endpoint:

```bash
export SERVICE_URL=https://<your-endpoint>.xenon.lepton.run
export EP_TOKEN=<endpoint token>
make observe                  # in this repo, starts Grafana on localhost:3000
```

### Replicas, terminal, logs

In the dashboard, open the endpoint → **Replicas** → for any replica:

- **Terminal** — opens a shell inside the container.
- **Logs** — live-tail of stdout/stderr.

For the full original click-through guide (with screenshots of each
section in the dashboard), see prior revisions of this file in git
history.
