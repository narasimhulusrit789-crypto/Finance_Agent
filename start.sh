#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  IBM Finance Agent — Linux/macOS Localhost Launcher
#  Model : IBM Granite-4.0-8B-Instruct
#  URL   : http://localhost:8000
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

echo ""
echo " ╔══════════════════════════════════════════════════════════╗"
echo " ║   IBM watsonx Finance Agent  —  Localhost               ║"
echo " ║   Model  : IBM Granite-4.0-8B-Instruct                  ║"
echo " ║   Project: d2be2c10-b24e-4054-b324-1e8fa543e0ad         ║"
echo " ║   Region : us-south.ml.cloud.ibm.com                    ║"
echo " ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── IBM watsonx.ai credentials ─────────────────────────────────
export WATSONX_API_KEY="aZP_dpBOLpoBH2MmXrODLAqSgzRN6UAG01XJ97VEkBz7"
export WATSONX_PROJECT_ID="d2be2c10-b24e-4054-b324-1e8fa543e0ad"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_REGION="us-south"

# ── Granite model ───────────────────────────────────────────────
export GRANITE_MODEL_ID="ibm/granite-4-h-small"
export GRANITE_MAX_TOKENS="4096"
export GRANITE_TEMPERATURE="0.7"
export GRANITE_TOP_P="0.95"

# ── IBM Cloud ────────────────────────────────────────────────────
export IBM_CLOUD_API_KEY="aZP_dpBOLpoBH2MmXrODLAqSgzRN6UAG01XJ97VEkBz7"
export IBM_CLOUD_REGION="us-south"
export IBM_CLOUD_RESOURCE_GROUP="default"

# ── Application ──────────────────────────────────────────────────
export APP_HOST="0.0.0.0"
export APP_PORT="8000"
export APP_ENV="production"
export SECRET_KEY="finance-agent-ibm-granite-4-secret-2025-x9k2"
export DEBUG="false"

# ── Vector / RAG ─────────────────────────────────────────────────
export VECTOR_DB_TYPE="chromadb"
export VECTOR_DB_PATH="./data/vector_store"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export CHUNK_SIZE="512"
export CHUNK_OVERLAP="64"
export TOP_K_RESULTS="5"

# ── Create required data directories ─────────────────────────────
mkdir -p data/vector_store data/uploads data/knowledge data/reports

echo " [1/3] Environment configured."
echo " [2/3] Data directories verified."
echo " [3/3] Starting server on http://localhost:8000 ..."
echo ""
echo " Dashboard : http://localhost:8000"
echo " API Docs  : http://localhost:8000/docs"
echo " Health    : http://localhost:8000/health"
echo ""
echo " Press Ctrl+C to stop."
echo ""

# Open browser in background after 4s
(sleep 4 && (xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || true)) &

python main.py
