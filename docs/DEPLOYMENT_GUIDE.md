# Deployment Guide

This guide covers deploying Planner to an OpenShift cluster using `deploy/kubernetes/deploy-all.sh`.

## Roadmap

The following improvements are planned to broaden platform and provider support:

- **Standard Kubernetes support** — Deploy on KIND, EKS, GKE, and other non-OpenShift clusters ([#96](https://github.com/llm-d-incubation/llm-d-planner/issues/96))
- ~~**OpenAI-compatible API support**~~ — Implemented. Set `LLM_PROVIDER=openai`. See LLM Provider section below.
- **Helm chart or Operator** — Replace the deploy script with a standard Kubernetes packaging format ([#223](https://github.com/llm-d-incubation/llm-d-planner/issues/223))

## Platform Support

| Platform | Status |
|----------|--------|
| OpenShift (cluster-admin) | Supported |
| OpenShift (namespace admin, no cluster-admin) | Supported |
| Standard Kubernetes (KIND, EKS, GKE, etc.) | Planned ([#96](https://github.com/llm-d-incubation/llm-d-planner/issues/96)) |

## Prerequisites

- `oc` CLI installed and authenticated (`oc login <cluster-url>`)
- Container images pushed to a registry accessible from the cluster

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `PLANNER_NAMESPACE` | `planner` | Target namespace. |
| `CLUSTER_ADMIN` | `true` | When `true`, creates the namespace and deploys ClusterRole/ClusterRoleBinding for GPU detection. Set to `false` when deploying to a pre-existing namespace where you don't have cluster-admin privileges. |
| `PLANNER_ENABLE_GPU` | `true` | When `false` and `LLM_PROVIDER=ollama`, removes `nvidia.com/gpu` resource requests from Ollama. |
| `DB_INIT` | `false` | When `true`, runs the `db-init` Job which loads pre-packaged simulated benchmark data (BLIS, interpolated, estimated) to simplify testing or demos. Optional — if not used, load your data separately via the web UI or API. |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` (local Ollama), `vertex` (Claude on Vertex AI), or `openai` (any OpenAI-compatible API). When not `ollama`, Ollama is not deployed. |
| `LLM_MODEL` | _(provider default)_ | Model name. Defaults: `qwen2.5:7b` (ollama), `claude-sonnet-4-6` (vertex), `gpt-4o` (openai). |
| `VERTEX_PROJECT_ID` | _(none)_ | GCP project ID. Only needed when `LLM_PROVIDER=vertex`. Injected into `planner-secrets`. |
| `VERTEX_REGION` | `global` | GCP region. Only needed when `LLM_PROVIDER=vertex`. |
| `GCP_CREDENTIALS_FILE` | `~/.config/gcloud/application_default_credentials.json` | Path to GCP ADC JSON file. Only used when `LLM_PROVIDER=vertex`. |
| `OPENAI_API_KEY` | _(none)_ | API key or proxy token. Required when `LLM_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | _(none)_ | Custom endpoint URL (e.g., LiteLLM proxy). Only needed when `LLM_PROVIDER=openai` with a non-default endpoint. |

### Secrets

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(none)_ | HuggingFace token for downloading gated models. Injected into the `planner-secrets` Secret. |
| `DB_ADMIN_PASSWORD` | _(none)_ | Optional password that locks the benchmark upload and reset API endpoints. When set, the UI shows a lock icon requiring the password before allowing database operations. Must be quoted if it contains special characters (e.g., `DB_ADMIN_PASSWORD='mypass!'`). |

## Deployment Scenarios

### OpenShift with Ollama (default)

Deploy to a dedicated `planner` namespace with Ollama as the LLM provider:

```bash
DB_INIT=true ./deploy/kubernetes/deploy-all.sh
```

This creates the `planner` namespace, deploys all components including Ollama with GPU, and initializes the database.

### OpenShift with Vertex AI (Claude)

Deploy using Claude on Vertex AI instead of Ollama:

```bash
LLM_PROVIDER=vertex \
VERTEX_PROJECT_ID=my-gcp-project \
DB_INIT=true \
./deploy/kubernetes/deploy-all.sh
```

This skips Ollama and configures the backend to use Claude via the Vertex AI API. Your local GCP Application Default Credentials are automatically patched into the cluster secret.

### OpenShift with OpenAI-Compatible API (e.g., LiteLLM)

Deploy using any OpenAI-compatible endpoint (LiteLLM, vLLM serving, etc.):

```bash
LLM_PROVIDER=openai \
LLM_MODEL=claude-sonnet-4-6 \
OPENAI_API_KEY=$LITELLM_TOKEN \
OPENAI_BASE_URL=$LITELLM_BASE_URL \
DB_INIT=true \
./deploy/kubernetes/deploy-all.sh
```

This skips Ollama and configures the backend to call the specified OpenAI-compatible endpoint.

### Pre-existing Namespace (no cluster-admin)

Deploy to a namespace you have access to but without cluster-admin privileges:

```bash
PLANNER_NAMESPACE=my-namespace \
CLUSTER_ADMIN=false \
LLM_PROVIDER=vertex \
VERTEX_PROJECT_ID=my-gcp-project \
HF_TOKEN=$HF_TOKEN \
DB_INIT=true \
./deploy/kubernetes/deploy-all.sh
```

When `CLUSTER_ADMIN=false`, the script skips namespace creation and ClusterRole/ClusterRoleBinding (used for GPU detection). A bare ServiceAccount is created instead.

### Subsequent Deploys (no DB init)

After the initial deploy, you can redeploy without reinitializing the database:

```bash
PLANNER_NAMESPACE=my-namespace \
CLUSTER_ADMIN=false \
LLM_PROVIDER=vertex \
VERTEX_PROJECT_ID=my-gcp-project \
./deploy/kubernetes/deploy-all.sh
```

### With Admin Password

Lock the benchmark database management endpoints behind a password:

```bash
DB_ADMIN_PASSWORD='secretpass' \
PLANNER_NAMESPACE=my-namespace \
CLUSTER_ADMIN=false \
LLM_PROVIDER=vertex \
VERTEX_PROJECT_ID=my-gcp-project \
DB_INIT=true \
./deploy/kubernetes/deploy-all.sh
```

## What the Script Does

1. **Copies** all YAML manifests from `deploy/kubernetes/` to a temporary `.rendered/` directory
2. **Transforms** the copies: injects namespace, secrets, configmap overrides via `sed`
3. **Applies** the rendered manifests to the cluster via `oc apply`
4. **Patches** GCP credentials into the secret (when using Vertex AI)
5. **Configures** Model Catalog NetworkPolicy (when `PLANNER_BENCHMARK_SOURCE=model_catalog`)
6. **Deploys** the backend (after prerequisites are ready)
7. **Initializes** the database (when `DB_INIT=true`)
8. **Cleans up** the `.rendered/` directory (which may contain injected secrets)

Source YAML files are never modified — all changes happen in the temporary rendered directory.

## Components Deployed

| Component | Manifest | Description |
|-----------|----------|-------------|
| Backend | `backend.yaml` | FastAPI API server |
| UI | `ui.yaml` | Streamlit web interface |
| Ollama | `ollama.yaml` | Local LLM server (only when `LLM_PROVIDER=ollama`) |
| Route | `route.yaml` | OpenShift Route for external access |

## Loading Benchmark Data

The database schema is automatically created on first data upload — no manual initialization required.

**Option 1: Via the UI** (recommended for manual setup)

1. Open the Planner UI (via the OpenShift Route)
2. Go to the **Configuration** tab
3. Upload a benchmark JSON file using the file uploader
4. Click **Load DB**

If `DB_ADMIN_PASSWORD` was set, click the lock icon and enter the password first.

**Option 2: Via `DB_INIT=true`** (for demos and testing)

Setting `DB_INIT=true` in the deploy command runs a Job that loads pre-packaged simulated benchmark data (`benchmarks_BLIS.json`, `benchmarks_interpolated_v2.json`, `benchmarks_estimated_performance.json`). Additional data can then be uploaded through the UI.

## Undeploying

To remove all Planner resources from a namespace (without deleting the namespace itself):

```bash
CLUSTER_ADMIN=false \
PLANNER_NAMESPACE=my-namespace \
./deploy/kubernetes/undeploy-all.sh
```

When `CLUSTER_ADMIN=true` (the default), the script also removes the ClusterRole and ClusterRoleBinding used for GPU detection.

## Troubleshooting

**Namespace creation forbidden**: You don't have cluster-admin privileges. Set `CLUSTER_ADMIN=false` to skip namespace creation and ClusterRole/ClusterRoleBinding, and set `PLANNER_NAMESPACE` to your pre-existing namespace.

**`Your default credentials were not found`**: GCP credentials weren't injected. Check that `GCP_CREDENTIALS_FILE` points to a valid credentials JSON file, or that your local ADC exists at `~/.config/gcloud/application_default_credentials.json`.
