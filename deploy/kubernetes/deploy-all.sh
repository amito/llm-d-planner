#!/bin/bash
set -e

NAMESPACE="${PLANNER_NAMESPACE:-planner}"
ENABLE_GPU="${PLANNER_ENABLE_GPU:-true}"
LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
CLUSTER_ADMIN="${CLUSTER_ADMIN:-true}"
DB_INIT="${DB_INIT:-false}"
RENDERED_DIR="deploy/kubernetes/.rendered"
trap "rm -rf \"$RENDERED_DIR\"" EXIT

echo "Deploying Planner to namespace: ${NAMESPACE}..."

# Prepare a clean rendered directory (copy source manifests, apply transforms there)
rm -rf "$RENDERED_DIR"
mkdir -p "$RENDERED_DIR"
cp deploy/kubernetes/*.yaml "$RENDERED_DIR/"

# Update namespace in all rendered manifests
for f in "$RENDERED_DIR"/*.yaml; do
  sed -i.bak "s/namespace: [a-zA-Z0-9_-]*/namespace: ${NAMESPACE}/g" "$f"
  rm -f "$f.bak"
done

# WARNING: sed-based injection breaks if values contain | & or newlines.
# This will be replaced by Kustomize overlays (see GitHub issue backlog).
# Inject HuggingFace token into secrets if provided via environment
if [ -n "$HF_TOKEN" ]; then
  sed -i.bak "s|hf-token: .*|hf-token: ${HF_TOKEN}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi

# Inject Vertex AI credentials into secrets if provided via environment
if [ -n "$VERTEX_PROJECT_ID" ]; then
  sed -i.bak "s|vertex-project-id: .*|vertex-project-id: ${VERTEX_PROJECT_ID}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi
if [ -n "$VERTEX_REGION" ]; then
  sed -i.bak "s|vertex-region: .*|vertex-region: ${VERTEX_REGION}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi

# Inject DB admin password into secrets if provided via environment
if [ -n "$DB_ADMIN_PASSWORD" ]; then
  sed -i.bak "s|db-admin-password: .*|db-admin-password: ${DB_ADMIN_PASSWORD}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi

# Inject OpenAI credentials into secrets if provided via environment
if [ -n "$OPENAI_API_KEY" ]; then
  sed -i.bak "s|openai-api-key: .*|openai-api-key: ${OPENAI_API_KEY}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi
if [ -n "$OPENAI_BASE_URL" ]; then
  sed -i.bak "s|openai-base-url: .*|openai-base-url: ${OPENAI_BASE_URL}|" "$RENDERED_DIR/secrets.yaml"
  rm -f "$RENDERED_DIR/secrets.yaml.bak"
fi

# Inject LLM_PROVIDER and LLM_MODEL into configmap
if [ "$LLM_PROVIDER" != "ollama" ]; then
  sed -i.bak "s|LLM_PROVIDER: .*|LLM_PROVIDER: ${LLM_PROVIDER}|" "$RENDERED_DIR/configmap.yaml"
  rm -f "$RENDERED_DIR/configmap.yaml.bak"
fi
if [ -n "$LLM_MODEL" ]; then
  sed -i.bak "s|LLM_MODEL: .*|LLM_MODEL: ${LLM_MODEL}|" "$RENDERED_DIR/configmap.yaml"
  rm -f "$RENDERED_DIR/configmap.yaml.bak"
fi

# Remove GPU request from ollama if disabled (only relevant when deploying Ollama)
if [ "$LLM_PROVIDER" = "ollama" ] && [ "$ENABLE_GPU" != "true" ]; then
  echo "GPU disabled, removing nvidia.com/gpu request from ollama..."
  sed -i.bak '/nvidia\.com\/gpu/d' "$RENDERED_DIR/ollama.yaml"
  rm -f "$RENDERED_DIR/ollama.yaml.bak"
fi

# Create namespace if cluster-admin (skip for pre-existing namespaces)
if [ "$CLUSTER_ADMIN" = "true" ]; then
  oc apply -f "$RENDERED_DIR/namespace.yaml"
fi

# Apply base infrastructure (everything except backend, which needs
# service-ca and NetworkPolicy to be ready first)
APPLY_ARGS="-f $RENDERED_DIR/secrets.yaml \
  -f $RENDERED_DIR/configmap.yaml \
  -f $RENDERED_DIR/service-ca-configmap.yaml \
  -f $RENDERED_DIR/postgres.yaml \
  -f $RENDERED_DIR/ui.yaml \
  -f $RENDERED_DIR/route.yaml"

# Only deploy Ollama when using it as the LLM provider
if [ "$LLM_PROVIDER" = "ollama" ]; then
  APPLY_ARGS="$APPLY_ARGS -f $RENDERED_DIR/ollama.yaml"
else
  echo "LLM_PROVIDER=${LLM_PROVIDER}, skipping Ollama deployment"
fi

# ClusterRole/ClusterRoleBinding require cluster-admin
if [ "$CLUSTER_ADMIN" = "true" ]; then
  APPLY_ARGS="$APPLY_ARGS -f $RENDERED_DIR/gpu-reader-rbac.yaml"
else
  echo "Skipping gpu-reader-rbac (CLUSTER_ADMIN=false)"
  echo "Creating planner-backend ServiceAccount..."
  oc apply -f - <<SAEOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: planner-backend
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: planner
SAEOF
fi

eval oc apply $APPLY_ARGS

# Patch GCP credentials into the secret after it's created
if [ "$LLM_PROVIDER" = "vertex" ]; then
  GCP_CREDS_FILE="${GCP_CREDENTIALS_FILE:-$HOME/.config/gcloud/application_default_credentials.json}"
  if [ -f "$GCP_CREDS_FILE" ]; then
    echo "Patching GCP credentials from $GCP_CREDS_FILE into planner-secrets..."
    oc patch secret planner-secrets -n "${NAMESPACE}" \
      -p "{\"stringData\":{\"gcp-credentials-json\":$(cat "$GCP_CREDS_FILE" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}}"
  else
    echo "Warning: GCP credentials file not found at $GCP_CREDS_FILE"
    echo "Set GCP_CREDENTIALS_FILE to the path of your credentials JSON"
  fi
fi

# Cross-namespace NetworkPolicy (allows planner backend -> Model Catalog)
BENCHMARK_SOURCE=$(oc get configmap planner-config -n "${NAMESPACE}" -o jsonpath='{.data.PLANNER_BENCHMARK_SOURCE}') || {
  echo "Warning: Failed to read planner-config configmap, skipping Model Catalog network policy"
  BENCHMARK_SOURCE=""
}
if [ "$BENCHMARK_SOURCE" = "model_catalog" ]; then
  echo "Applying Model Catalog network policy..."
  oc apply -f "$RENDERED_DIR/networkpolicy-model-catalog.yaml"

  echo "Waiting for service-ca certificate injection..."
  for i in $(seq 1 30); do
    if oc get configmap planner-service-ca -n "${NAMESPACE}" -o jsonpath='{.data.service-ca\.crt}' 2>/dev/null | grep -q "BEGIN CERTIFICATE"; then
      echo "Service CA certificate is ready."
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "Error: Timed out waiting for service-ca certificate injection" >&2
      exit 1
    fi
    sleep 2
  done
else
  echo "Skipping Model Catalog network policy (benchmark source: ${BENCHMARK_SOURCE:-postgresql})"
  if [ "$CLUSTER_ADMIN" = "true" ]; then
    oc delete -f "$RENDERED_DIR/networkpolicy-model-catalog.yaml" --ignore-not-found
  fi
fi

# Apply backend after prerequisites are ready
echo "Deploying backend..."
oc apply -f "$RENDERED_DIR/backend.yaml"

if [ "$DB_INIT" = "true" ]; then
  echo "Waiting for PostgreSQL to be ready..."
  oc wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n "${NAMESPACE}" --timeout=120s

  echo "Running database initialization job..."
  # Delete previous job if it exists (jobs are immutable)
  oc delete job db-init -n "${NAMESPACE}" --ignore-not-found
  oc apply -f "$RENDERED_DIR/db-init-job.yaml"

  echo "Waiting for db-init job to complete..."
  oc wait --for=condition=complete job/db-init -n "${NAMESPACE}" --timeout=300s

  echo "Database initialized."
else
  echo "Skipping database initialization (DB_INIT=${DB_INIT})"
fi

echo "Deployment complete."
