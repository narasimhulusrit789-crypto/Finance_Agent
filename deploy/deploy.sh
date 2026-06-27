#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  IBM Code Engine — Finance Agent (Granite-4.0-8B-Instruct) Deploy
#  Project ID : d2be2c10-b24e-4054-b324-1e8fa543e0ad
#  Region     : us-south
#  Endpoint   : https://us-south.ml.cloud.ibm.com
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
REGION="us-south"
RESOURCE_GROUP="default"
CE_PROJECT_NAME="finance-agent-project"
APP_NAME="finance-agent"
NAMESPACE="finance-agent-ns"
REGISTRY_NS="${NAMESPACE}"
IMAGE_TAG="latest"
IMAGE="us.icr.io/${REGISTRY_NS}/${APP_NAME}:${IMAGE_TAG}"

WATSONX_API_KEY="aZP_dpBOLpoBH2MmXrODLAqSgzRN6UAG01XJ97VEkBz7"
WATSONX_PROJECT_ID="d2be2c10-b24e-4054-b324-1e8fa543e0ad"
WATSONX_URL="https://us-south.ml.cloud.ibm.com"
GRANITE_MODEL_ID="ibm/granite-4-h-small"
APP_SECRET_KEY="finance-agent-ibm-granite-4-secret-2025-x9k2"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   IBM Finance Agent — Cloud Deployment                      ║"
echo "║   Model : IBM Granite-4.0-8B-Instruct                       ║"
echo "║   Region: us-south                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1 — Login ────────────────────────────────────────────────
echo "[1/7] Logging in to IBM Cloud..."
ibmcloud login --apikey "${WATSONX_API_KEY}" -r "${REGION}" -g "${RESOURCE_GROUP}"

# ── Step 2 — Container Registry ──────────────────────────────────
echo "[2/7] Logging in to IBM Container Registry..."
ibmcloud cr login
ibmcloud cr namespace-add "${REGISTRY_NS}" 2>/dev/null || echo "  Namespace already exists, continuing."

# ── Step 3 — Build & Push Docker Image ───────────────────────────
echo "[3/7] Building Docker image..."
docker build -t "${IMAGE}" .

echo "[3/7] Pushing image to IBM Container Registry..."
docker push "${IMAGE}"

# ── Step 4 — Code Engine Project ─────────────────────────────────
echo "[4/7] Selecting / creating IBM Code Engine project..."
ibmcloud ce project select --name "${CE_PROJECT_NAME}" 2>/dev/null || \
  ibmcloud ce project create --name "${CE_PROJECT_NAME}" --region "${REGION}"

# ── Step 5 — Secrets ─────────────────────────────────────────────
echo "[5/7] Creating / updating secrets..."
ibmcloud ce secret delete --name finance-agent-secrets --force 2>/dev/null || true
ibmcloud ce secret create --name finance-agent-secrets \
  --from-literal WATSONX_API_KEY="${WATSONX_API_KEY}" \
  --from-literal WATSONX_PROJECT_ID="${WATSONX_PROJECT_ID}" \
  --from-literal SECRET_KEY="${APP_SECRET_KEY}"

# ── Step 6 — Deploy Application ──────────────────────────────────
echo "[6/7] Deploying application to IBM Code Engine..."
if ibmcloud ce application get --name "${APP_NAME}" &>/dev/null; then
  echo "  Updating existing application..."
  ibmcloud ce application update \
    --name "${APP_NAME}" \
    --image "${IMAGE}" \
    --cpu 2 \
    --memory 4G \
    --min-scale 1 \
    --max-scale 5 \
    --concurrency 20 \
    --timeout 300 \
    --port 8000 \
    --env APP_ENV=production \
    --env APP_HOST=0.0.0.0 \
    --env APP_PORT=8000 \
    --env WATSONX_URL="${WATSONX_URL}" \
    --env WATSONX_REGION="${REGION}" \
    --env GRANITE_MODEL_ID="${GRANITE_MODEL_ID}" \
    --env GRANITE_MAX_TOKENS=4096 \
    --env GRANITE_TEMPERATURE=0.7 \
    --env GRANITE_TOP_P=0.95 \
    --env IBM_CLOUD_REGION="${REGION}" \
    --env IBM_CLOUD_RESOURCE_GROUP="${RESOURCE_GROUP}" \
    --env VECTOR_DB_TYPE=chromadb \
    --env VECTOR_DB_PATH=./data/vector_store \
    --env EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    --env CHUNK_SIZE=512 \
    --env CHUNK_OVERLAP=64 \
    --env TOP_K_RESULTS=5 \
    --env DEBUG=false \
    --env-from-secret finance-agent-secrets
else
  echo "  Creating new application..."
  ibmcloud ce application create \
    --name "${APP_NAME}" \
    --image "${IMAGE}" \
    --cpu 2 \
    --memory 4G \
    --min-scale 1 \
    --max-scale 5 \
    --concurrency 20 \
    --timeout 300 \
    --port 8000 \
    --env APP_ENV=production \
    --env APP_HOST=0.0.0.0 \
    --env APP_PORT=8000 \
    --env WATSONX_URL="${WATSONX_URL}" \
    --env WATSONX_REGION="${REGION}" \
    --env GRANITE_MODEL_ID="${GRANITE_MODEL_ID}" \
    --env GRANITE_MAX_TOKENS=4096 \
    --env GRANITE_TEMPERATURE=0.7 \
    --env GRANITE_TOP_P=0.95 \
    --env IBM_CLOUD_REGION="${REGION}" \
    --env IBM_CLOUD_RESOURCE_GROUP="${RESOURCE_GROUP}" \
    --env VECTOR_DB_TYPE=chromadb \
    --env VECTOR_DB_PATH=./data/vector_store \
    --env EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    --env CHUNK_SIZE=512 \
    --env CHUNK_OVERLAP=64 \
    --env TOP_K_RESULTS=5 \
    --env DEBUG=false \
    --env-from-secret finance-agent-secrets
fi

# ── Step 7 — Get URL ─────────────────────────────────────────────
echo "[7/7] Fetching deployed application URL..."
APP_URL=$(ibmcloud ce application get --name "${APP_NAME}" --output url 2>/dev/null || echo "pending")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Deployment Complete!                                       ║"
echo "║   App URL : ${APP_URL}"
echo "║   Health  : ${APP_URL}/health                               ║"
echo "║   API     : ${APP_URL}/api/chat                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
