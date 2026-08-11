# PhysicsNeMo Serve — Azure ML Deployment Guide

Deploy the `physicsnemo-serve` Docker image to an Azure ML managed online endpoint
and call REST APIs against it.

---

## Environment Variables

Set these once in your shell before running any commands in the phases below.
Every command in this guide uses these variables — no manual substitution needed.

```bash
# ── Azure identity ──────────────────────────────────────────────────────────
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
export AZURE_RESOURCE_GROUP="<your-resource-group>"
export AZURE_ML_WORKSPACE="<your-workspace-name>"

# ── Azure location ───────────────────────────────────────────────────────────
export LOCATION="<azure-region>"          # e.g. eastus, westus3

# ── Container registry ───────────────────────────────────────────────────────
export ACR_NAME="<your-acr-name>"
export ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# ── Image names ──────────────────────────────────────────────────────────────
export RUNTIME_BASE_IMAGE="${ACR_LOGIN_SERVER}/physicsnemo-serve-runtime-base:pytorch-26.01-py3-th0.8.0"
export IMAGE_NAME="${ACR_LOGIN_SERVER}/physicsnemo-serve"
export IMAGE_TAG="$(git rev-parse --short HEAD)"   # run from the repo root

# ── Azure ML endpoint / deployment ───────────────────────────────────────────
export ENDPOINT_NAME="pnserve-deterministic"
export DEPLOYMENT_NAME="pnserve-deploy"
export PLUGIN_ID="e2s-deterministic"
export INSTANCE_TYPE="Standard_NC40ads_H100_v5"   # 1x H100 GPU

# ── NGC (only when building from NVIDIA's NGC PyTorch base image) ────────────
# export NGC_API_KEY="..."   # not needed with a base image from another registry
```

> **Tip:** Save the block above to a file like `~/.azure-pnserve.env` and
> `source ~/.azure-pnserve.env` at the start of each session.

---

## Phase 0 — Install Tooling (one-time)

### Step 1 — Install Azure CLI

```bash
# Debian/Ubuntu
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Verify:
```bash
az --version
```

### Step 2 — Add the Azure ML extension

```bash
az extension add --name ml --upgrade
```

### Step 3 — Log in and set defaults

```bash
az login    # opens a browser window; complete the sign-in there

az account set --subscription "$AZURE_SUBSCRIPTION_ID"

az configure \
  --defaults group="$AZURE_RESOURCE_GROUP" \
             workspace="$AZURE_ML_WORKSPACE" \
             location="$LOCATION"

# Verify:
az account show --query "{name:name, id:id}" -o table
az ml workspace show --query "{name:name, location:location}" -o table
```

---

## Phase 1 — Find the ACR Name

### Step 1 — List ACR registries in your resource group

```bash
az acr list --resource-group "$AZURE_RESOURCE_GROUP" -o table
```

You'll see output like:
```
NAME       RESOURCE GROUP      LOCATION    SKU      LOGIN SERVER
myacr      my-resource-group   eastus      Premium  myacr.azurecr.io
```

### Step 2 — Set ACR_NAME and derived variables

Once you know the name, update your env:
```bash
export ACR_NAME="<name-from-table-above>"
export ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"
export RUNTIME_BASE_IMAGE="${ACR_LOGIN_SERVER}/physicsnemo-serve-runtime-base:pytorch-26.01-py3-th0.8.0"
export IMAGE_NAME="${ACR_LOGIN_SERVER}/physicsnemo-serve"
```

### Step 3 — Authenticate Docker to ACR

```bash
az acr login --name "$ACR_NAME"
```

---

## Phase 2 — Build and Push Images

Run all commands from the root of your `physicsnemo-serve` checkout.

### Step 1 — Check for the runtime-base image

This image changes rarely (only when the PyTorch version or system dependencies change).
Check if it already exists in ACR before building:

```bash
az acr repository show --name "$ACR_NAME" \
  --image "physicsnemo-serve-runtime-base:pytorch-26.01-py3-th0.8.0" 2>/dev/null \
  && echo "Runtime base already in ACR — skip build" \
  || echo "Not found — need to build"
```

If it exists, skip Step 2. You do not need an NGC API key unless you need to build
the runtime-base image from NVIDIA's NGC PyTorch image.

### Step 2 — Build and push the runtime-base image (only if missing)

The command below uses NVIDIA's official PyTorch image from NGC. Authenticate to
`nvcr.io` before building:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Then build and push the runtime-base image (takes ~15–20 min on first run):

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 \
  -t "$RUNTIME_BASE_IMAGE" \
  -f Dockerfile.physicsnemo-serve.runtime-base .

docker push "$RUNTIME_BASE_IMAGE"
```

#### Alternatives to NGC

NGC is not required by PhysicsNeMo Serve itself. It is used here only to pull the
example PyTorch base image. To use a PyTorch image from another registry, set its
reference and pass it through the Dockerfile's existing `PYTORCH_BASE_IMAGE` build
argument—no Dockerfile edit is required:

```bash
export PYTORCH_BASE_IMAGE="<registry>/<repository>:<tag>"

DOCKER_BUILDKIT=1 docker build \
  --build-arg PYTORCH_BASE_IMAGE="$PYTORCH_BASE_IMAGE" \
  -t "$RUNTIME_BASE_IMAGE" \
  -f Dockerfile.physicsnemo-serve.runtime-base .

docker push "$RUNTIME_BASE_IMAGE"
```

Log in to the alternative registry only if it requires authentication. A public
image can be pulled without an NGC API key. The alternative image must be compatible
with `Dockerfile.physicsnemo-serve.runtime-base`: Linux/amd64, an `apt`-based userland,
Python 3, a CUDA-enabled PyTorch installation, and a CUDA toolkit available at
`/usr/local/cuda` for building CUDA extensions.

If you already have a compatible prebuilt PhysicsNeMo Serve runtime-base image,
set `RUNTIME_BASE_IMAGE` to that image and skip this step entirely. Authenticate to
its registry if necessary.

### Step 3 — Build and push the main service image

The Makefile reads `DOCKER_REPO` and `IMAGE_NAME` from env vars:

```bash
DOCKER_BUILDKIT=1 DOCKER_REPO="$ACR_LOGIN_SERVER" \
  docker build \
  --build-arg PHYSICSNEMO_SERVE_RUNTIME_BASE_IMAGE="$RUNTIME_BASE_IMAGE" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f Dockerfile.physicsnemo-serve.scicomp-rust-slim .

docker push "${IMAGE_NAME}:${IMAGE_TAG}"
```

Confirm the image is in ACR:
```bash
az acr repository show-tags --name "$ACR_NAME" --repository physicsnemo-serve -o table
```

---

## Phase 3 — Create Deployment YAML Files

Create a local directory to hold your deployment assets (outside the repo is fine):
```bash
mkdir -p ~/pnserve-azure
```

### endpoint.yml

```bash
cat > ~/pnserve-azure/endpoint.yml << EOF
\$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: ${ENDPOINT_NAME}
auth_mode: key
EOF
```

### deployment.yml

```bash
cat > ~/pnserve-azure/deployment.yml << EOF
\$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: ${DEPLOYMENT_NAME}
endpoint_name: ${ENDPOINT_NAME}
environment_variables:
  SERVER_PORT: "8080"
  PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID: "${PLUGIN_ID}"
  HEALTH_STUB_ENABLED: "false"
environment:
  name: pnserve-env
  image: ${IMAGE_NAME}:${IMAGE_TAG}
  inference_config:
    liveness_route:
      port: 8080
      path: /healthz
    readiness_route:
      port: 8080
      path: /readyz
    scoring_route:
      port: 8080
      path: /v1/infer/${PLUGIN_ID}/run
instance_type: ${INSTANCE_TYPE}
instance_count: 1
request_settings:
  request_timeout_ms: 90000
EOF
```

> **Note on routes:** physicsnemo-serve's Rust server exposes `/healthz` (liveness),
> `/readyz` (readiness), and `/v1/infer/<plugin-id>/run` (scoring). Azure ML's
> `liveness_route` / `readiness_route` / `scoring_route` fields are just configuration —
> they can point to any path the container exposes. No source code changes are needed.

> **Note on model weights:** The `e2s-deterministic` plugin downloads model weights
> at runtime. The first inference request will be slower while weights are downloaded
> (~a few minutes); subsequent requests within the same deployment lifetime are fast.

> **Note on `request_timeout_ms`:** Azure ML's default HTTP timeout is 5 seconds,
> which is too short for inference. The value above sets it to 90 seconds. Adjust
> upward if your plugin has longer inference times.

---

## Phase 4 — Deploy to Azure ML

### Step 1 — Create the endpoint

Registers the endpoint and creates its managed identity. No VM is allocated yet.

```bash
az ml online-endpoint create -f ~/pnserve-azure/endpoint.yml \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP"
```

### Step 2 — Grant ACR pull access to the endpoint's managed identity

Azure ML uses the endpoint's managed identity to pull your image from ACR.
This role assignment must exist before the deployment can start.

```bash
PRINCIPAL_ID="$(az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query identity.principal_id -o tsv)"

ACR_SCOPE="$(az acr show --name "$ACR_NAME" --query id -o tsv)"

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "AcrPull" \
  --scope "$ACR_SCOPE"
```

> RBAC changes can take 2–3 minutes to propagate. If the deployment fails immediately
> after this step, wait and retry.

### Step 3 — Create the deployment

Allocates the GPU VM, pulls the image, and starts the service. Takes 10–20 minutes.

```bash
az ml online-deployment create -f ~/pnserve-azure/deployment.yml \
  --all-traffic \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP"
```

### Step 4 — Monitor deployment logs

```bash
az ml online-deployment get-logs \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --lines 200
```

### Step 5 — Confirm the deployment is healthy

```bash
az ml online-deployment show \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query "{state:provisioning_state, instance_type:instance_type}" \
  -o table
```

---

## Phase 5 — Call REST APIs

### Step 1 — Get the scoring URI and auth key

```bash
SCORING_URI="$(az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query scoring_uri -o tsv)"

KEY="$(az ml online-endpoint get-credentials \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query primaryKey -o tsv)"

# Strip the scoring route suffix to get the API base URL
API_BASE="${SCORING_URI%/v1/infer/*}"

echo "API base: $API_BASE"
# Expected: https://<endpoint-name>.<region>.inference.ml.azure.com
```

> Use `$KEY` as the Bearer token in all curl calls below.

### Step 2 — Health check

```bash
curl -sS "$API_BASE/healthz" \
  -H "Authorization: Bearer $KEY"
```

> Returns plain text `ok`, not JSON — do not pipe through `jq`.

### Step 3 — List available plugins

```bash
curl -sS "$API_BASE/v1/infer/workflows" \
  -H "Authorization: Bearer $KEY" | jq
```

### Step 4 — Inspect the plugin's input schema

```bash
curl -sS "$API_BASE/v1/infer/${PLUGIN_ID}/schema" \
  -H "Authorization: Bearer $KEY" | jq
```

### Step 5 — Check plugin readiness

```bash
curl -sS "$API_BASE/v1/infer/${PLUGIN_ID}/readiness" \
  -H "Authorization: Bearer $KEY" | jq
```

### Step 6 — Submit an inference run

```bash
curl -sS -X POST "$API_BASE/v1/infer/${PLUGIN_ID}/run" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d @plugins/${PLUGIN_ID}/examples/default_request.json
```

The response contains a `run_id`. Set it:
```bash
RUN_ID="<run_id from the response above>"
```

### Step 7 — Poll run status

```bash
curl -sS "$API_BASE/v1/infer/${PLUGIN_ID}/${RUN_ID}/status" \
  -H "Authorization: Bearer $KEY" | jq
```

### Step 8 — Fetch results

```bash
curl -sS "$API_BASE/v1/infer/${PLUGIN_ID}/${RUN_ID}/results" \
  -H "Authorization: Bearer $KEY" | jq
```

### Interactive API docs (Swagger UI)

```bash
echo "$API_BASE/doc"
```
Open that URL in your browser and paste `$KEY` as the Bearer token in the Authorize dialog.

---

## Operations Reference

### Check GPU quota for a region

```bash
az vm list-usage --location "$LOCATION" -o table | grep -i "NC\|ND"
```

### Update the deployment with a new image

```bash
# After building and pushing a new image tag:
export IMAGE_TAG="<new-tag>"

# Regenerate deployment.yml (Phase 3) then:
az ml online-deployment update \
  -f ~/pnserve-azure/deployment.yml \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP"
```

### Tear down (stops billing immediately)

Delete the deployment first (required if it has active traffic), then the endpoint:

```bash
# Zero out traffic if the deployment is healthy
az ml online-endpoint update \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --traffic "${DEPLOYMENT_NAME}=0"

az ml online-deployment delete \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --yes

az ml online-endpoint delete \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --yes
```

---

## Troubleshooting

### Step 1 — Check deployment state

```bash
az ml online-deployment show \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query "{state:provisioning_state, instance_type:instance_type}" \
  -o table
```

### Step 2 — Get container logs

```bash
az ml online-deployment get-logs \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --lines 200
```

If logs are empty, the failure happened before the container started (infrastructure or
capacity issue) — continue to Step 3.

### Step 3 — Get the real error from the async operation

When logs are empty and `provisioning_state` is `Failed`, call the Azure async operation
URL directly to get the underlying error:

```bash
OP_URL=$(az ml online-deployment show \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  -o json | jq -r '.properties.AzureAsyncOperationUri')

az rest --method GET --url "$OP_URL"
```

Common error codes:

| Code | Cause | Fix |
|---|---|---|
| `InternalServerError` at `percentComplete: 0.0` | Missing AcrPull role **or** GPU capacity exhausted | Check roles first (Step 4) — if roles are fine, retry at off-peak hours or try a different region |
| `ImagePullBackOff` / `AuthorizationFailed` | Endpoint managed identity missing AcrPull role | Assign AcrPull role (Step 4) |
| `CapacityError` / `AllocationFailed` | No quota or capacity for the requested SKU | Try a different instance type or region; contact your Azure admin to increase quota |

> **Note:** `InternalServerError` at `percentComplete: 0.0` is ambiguous — Azure returns
> the same generic error for both a missing AcrPull role and a capacity failure. Always
> check roles first (Step 4) before concluding it is a capacity issue.

### Step 4 — Check and fix the endpoint's managed identity roles

```bash
PRINCIPAL_ID="$(az ml online-endpoint show \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query identity.principal_id -o tsv)"

# List all role assignments for the endpoint identity
az role assignment list --assignee "$PRINCIPAL_ID" -o table
```

You should see at minimum an `AcrPull` role scoped to your ACR.
If the list is empty or `AcrPull` is missing, assign it:

```bash
ACR_SCOPE="$(az acr show --name "$ACR_NAME" --query id -o tsv)"

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "AcrPull" \
  --scope "$ACR_SCOPE"
```

Wait 2–3 minutes for RBAC to propagate, then delete the failed deployment and retry (Step 5).

### Step 5 — Delete a failed deployment and retry

The endpoint itself can stay — only the deployment needs to be deleted and recreated:

```bash
az ml online-deployment delete \
  --name "$DEPLOYMENT_NAME" \
  --endpoint-name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --yes

az ml online-deployment create \
  -f ~/pnserve-azure/deployment.yml \
  --all-traffic \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP"
```

### Step 6 — Delete everything and start fresh

Only needed if the endpoint itself is stuck in a bad state. Deletes the endpoint and
all deployments under it. Billing stops immediately.

```bash
az ml online-endpoint delete \
  --name "$ENDPOINT_NAME" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --yes
```

Then recreate from Phase 4, Step 1.
