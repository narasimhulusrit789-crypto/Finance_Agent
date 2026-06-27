@echo off
REM ═══════════════════════════════════════════════════════════════
REM  IBM Finance Agent — Windows Localhost Launcher
REM  Model : IBM Granite-4.0-8B-Instruct
REM  URL   : http://localhost:8000
REM ═══════════════════════════════════════════════════════════════
title IBM watsonx Finance Agent

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   IBM watsonx Finance Agent  —  Localhost               ║
echo  ║   Model  : IBM Granite-4.0-8B-Instruct                  ║
echo  ║   Project: d2be2c10-b24e-4054-b324-1e8fa543e0ad         ║
echo  ║   Region : us-south.ml.cloud.ibm.com                    ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM ── IBM watsonx.ai credentials ─────────────────────────────
set WATSONX_API_KEY=aZP_dpBOLpoBH2MmXrODLAqSgzRN6UAG01XJ97VEkBz7
set WATSONX_PROJECT_ID=d2be2c10-b24e-4054-b324-1e8fa543e0ad
set WATSONX_URL=https://us-south.ml.cloud.ibm.com
set WATSONX_REGION=us-south

REM ── Granite model ──────────────────────────────────────────
set GRANITE_MODEL_ID=ibm/granite-4-h-small
set GRANITE_MAX_TOKENS=4096
set GRANITE_TEMPERATURE=0.7
set GRANITE_TOP_P=0.95

REM ── IBM Cloud ───────────────────────────────────────────────
set IBM_CLOUD_API_KEY=aZP_dpBOLpoBH2MmXrODLAqSgzRN6UAG01XJ97VEkBz7
set IBM_CLOUD_REGION=us-south
set IBM_CLOUD_RESOURCE_GROUP=default

REM ── Application ─────────────────────────────────────────────
set APP_HOST=0.0.0.0
set APP_PORT=8000
set APP_ENV=production
set SECRET_KEY=finance-agent-ibm-granite-4-secret-2025-x9k2
set DEBUG=false

REM ── Vector / RAG ────────────────────────────────────────────
set VECTOR_DB_TYPE=chromadb
set VECTOR_DB_PATH=./data/vector_store
set EMBEDDING_MODEL=all-MiniLM-L6-v2
set CHUNK_SIZE=512
set CHUNK_OVERLAP=64
set TOP_K_RESULTS=5

REM ── Create required data directories ────────────────────────
if not exist "data\vector_store" mkdir "data\vector_store"
if not exist "data\uploads"      mkdir "data\uploads"
if not exist "data\knowledge"    mkdir "data\knowledge"
if not exist "data\reports"      mkdir "data\reports"

echo  [1/3] Environment configured.
echo  [2/3] Data directories verified.
echo  [3/3] Starting server on http://localhost:8000 ...
echo.
echo  Dashboard : http://localhost:8000
echo  API Docs  : http://localhost:8000/docs
echo  Health    : http://localhost:8000/health
echo.
echo  Press Ctrl+C to stop.
echo.

REM Open browser after 4 seconds in background
start /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8000"

python main.py
pause
