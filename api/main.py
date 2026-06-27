"""
FastAPI Application — Main entry point
────────────────────────────────────────────────────────────────
Provides:
  • /api/chat          — Main chat + multi-agent pipeline
  • /api/upload        — Document upload → RAG ingestion
  • /api/portfolio     — Portfolio CRUD
  • /api/market        — Live market data endpoint
  • /api/sessions      — Session management
  • /ws/chat/{sid}     — WebSocket for real-time streaming
  • /                  — Serve dashboard (static HTML)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.models import PortfolioHolding, RiskLevel, UserProfile
from agents.orchestrator import AgentOrchestrator
from rag.pipeline import RAGIngestionPipeline, RAGRetriever, VectorStore
from storage.file_component import FileComponent

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────
#  Application lifecycle
# ─────────────────────────────────────────────────────────────

_vector_store: VectorStore | None = None
_retriever: RAGRetriever | None = None
_orchestrator: AgentOrchestrator | None = None
_file_component: FileComponent | None = None
_ingestion_pipeline: RAGIngestionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy singletons once at startup."""
    global _vector_store, _retriever, _orchestrator, _file_component, _ingestion_pipeline

    logger.info("app.startup")
    _vector_store = VectorStore()
    _retriever = RAGRetriever(_vector_store)
    _orchestrator = AgentOrchestrator(retriever=_retriever)
    _file_component = FileComponent()
    _ingestion_pipeline = RAGIngestionPipeline(_vector_store)

    # Pre-ingest any documents in the default knowledge directory
    knowledge_dir = os.path.join("data", "knowledge")
    if os.path.isdir(knowledge_dir):
        results = _ingestion_pipeline.ingest_directory(knowledge_dir)
        logger.info("app.pre_ingestion", files=len(results), chunks=sum(v for v in results.values() if v > 0))

    logger.info("app.ready", doc_count=_vector_store.count())
    yield
    logger.info("app.shutdown")


# ─────────────────────────────────────────────────────────────
#  FastAPI application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="IBM watsonx Finance Agent",
    description="AI-Powered Investment Recommendation System — IBM BOB Multi-Agent Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend/index.html as root dashboard
_frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─────────────────────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's investment query")
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str | None = None
    investment_goal: str = Field("Wealth Growth")
    budget: float = Field(10000.0, gt=0)
    risk_level: RiskLevel = Field(RiskLevel.MEDIUM)
    investment_horizon_years: int = Field(10, ge=1, le=50)
    monthly_contribution: float = Field(0.0, ge=0)
    age: int | None = None
    existing_portfolio: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    message: str
    recommendation: dict | None = None
    risk_metrics: dict | None = None
    market_insight: dict | None = None
    sources: list[str] = []
    knowledge_summary: str = ""


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int
    total_documents: int
    storage_key: str


class PortfolioSaveRequest(BaseModel):
    user_id: str
    portfolio: dict


# ─────────────────────────────────────────────────────────────
#  Chat endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint — triggers the full multi-agent pipeline.
    """
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")

    # Build holdings from raw dicts
    holdings = []
    for h in req.existing_portfolio:
        try:
            holdings.append(PortfolioHolding(**h))
        except Exception:  # noqa: BLE001
            pass

    user_profile = UserProfile(
        user_id=req.user_id,
        investment_goal=req.investment_goal,
        budget=req.budget,
        risk_level=req.risk_level,
        investment_horizon_years=req.investment_horizon_years,
        monthly_contribution=req.monthly_contribution,
        age=req.age,
        existing_portfolio=holdings,
    )

    try:
        output = await _orchestrator.run(
            user_input=req.message,
            user_profile=user_profile,
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error("chat.pipeline_error", error=str(exc))
        raise HTTPException(500, f"Agent pipeline error: {exc}") from exc

    # Save report to file storage
    if _file_component:
        try:
            _file_component.save_report(
                "investment_recommendation",
                output.model_dump(),
                req.user_id,
            )
        except Exception:  # noqa: BLE001
            pass

    return ChatResponse(
        session_id=output.session_id,
        user_id=output.user_id,
        message=output.final_narrative,
        recommendation=output.recommendation.model_dump() if output.recommendation else None,
        risk_metrics=output.risk_metrics.model_dump() if output.risk_metrics else None,
        market_insight=output.market_insight.model_dump() if output.market_insight else None,
        sources=output.sources,
        knowledge_summary=output.knowledge_summary,
    )


# ─────────────────────────────────────────────────────────────
#  Document upload + RAG ingestion
# ─────────────────────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    user_id: str = "anonymous",
    file: UploadFile = File(...),
):
    """Upload a financial document for RAG ingestion."""
    if _ingestion_pipeline is None or _file_component is None:
        raise HTTPException(503, "Service not ready")

    allowed_types = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".md"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed_types:
        raise HTTPException(400, f"File type {suffix} not supported. Allowed: {allowed_types}")

    content = await file.read()

    # Store in file component (COS or local)
    storage_key = _file_component.upload_user_document(user_id, file.filename or "upload", content)

    # Ingest into vector store
    try:
        chunks = _ingestion_pipeline.ingest_bytes(content, file.filename or "upload")
    except Exception as exc:
        raise HTTPException(500, f"RAG ingestion failed: {exc}") from exc

    return UploadResponse(
        filename=file.filename or "upload",
        chunks_ingested=chunks,
        total_documents=_ingestion_pipeline.document_count,
        storage_key=storage_key,
    )


# ─────────────────────────────────────────────────────────────
#  Portfolio endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/api/portfolio/save")
async def save_portfolio(req: PortfolioSaveRequest):
    if _file_component is None:
        raise HTTPException(503, "Service not ready")
    key = _file_component.save_portfolio(req.user_id, req.portfolio)
    return {"status": "saved", "key": key}


@app.get("/api/portfolio/{user_id}")
async def get_portfolio(user_id: str):
    if _file_component is None:
        raise HTTPException(503, "Service not ready")
    portfolio = _file_component.load_portfolio(user_id)
    if not portfolio:
        raise HTTPException(404, f"No portfolio found for user {user_id}")
    return portfolio


@app.get("/api/portfolio/{user_id}/files")
async def list_portfolio_files(user_id: str):
    if _file_component is None:
        raise HTTPException(503, "Service not ready")
    files = _file_component.list_files(f"portfolios/{user_id}/")
    return {"files": files, "count": len(files)}


# ─────────────────────────────────────────────────────────────
#  Market data endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/market/snapshot")
async def market_snapshot():
    """Return a real-time market snapshot (uses MarketMonitoringAgent)."""
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")
    # Trigger market agent standalone
    from agents.market_monitoring_agent import MarketMonitoringAgent
    from agents.watsonx_client import GraniteClient
    agent = MarketMonitoringAgent(_orchestrator._llm)
    dummy_profile = {"risk_level": "Medium", "investment_goal": "General"}
    resp = await agent.execute("market_snapshot", {"user_profile": dummy_profile})
    return resp.result.get("market_data_snapshot", {})


# ─────────────────────────────────────────────────────────────
#  Session management
# ─────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str):
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")
    history = _orchestrator.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")
    _orchestrator.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")
    return {"sessions": _orchestrator.active_sessions}


# ─────────────────────────────────────────────────────────────
#  Agent message log
# ─────────────────────────────────────────────────────────────

@app.get("/api/agents/message-log")
async def agent_message_log():
    if _orchestrator is None:
        raise HTTPException(503, "Service not ready")
    return {"messages": _orchestrator.get_message_log()}


# ─────────────────────────────────────────────────────────────
#  Vector store stats
# ─────────────────────────────────────────────────────────────

@app.get("/api/rag/stats")
async def rag_stats():
    if _vector_store is None:
        raise HTTPException(503, "Service not ready")
    return {
        "total_documents": _vector_store.count(),
        "backend": _vector_store._backend,
    }


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  Root — serve dashboard UI
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the Finance Agent dashboard UI."""
    if os.path.isfile(_frontend_path):
        with open(_frontend_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>IBM watsonx Finance Agent</h1><p>Frontend not found. Run the server from the project root.</p>")


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "orchestrator": _orchestrator is not None,
        "vector_store": _vector_store.count() if _vector_store else 0,
        "file_component": _file_component._backend if _file_component else "unavailable",
    }


# ─────────────────────────────────────────────────────────────
#  WebSocket — streaming chat
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming responses.
    Sends agent progress events as JSON frames.
    """
    await websocket.accept()
    logger.info("ws.connected", session_id=session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if _orchestrator is None:
                await websocket.send_json({"error": "Service not ready"})
                continue

            # Send progress notifications as the pipeline runs
            await websocket.send_json({"event": "progress", "stage": "knowledge_agent", "message": "Searching financial knowledge base..."})

            try:
                holdings = [PortfolioHolding(**h) for h in data.get("existing_portfolio", []) if isinstance(h, dict)]
                profile = UserProfile(
                    user_id=data.get("user_id", "ws_user"),
                    investment_goal=data.get("investment_goal", "Wealth Growth"),
                    budget=float(data.get("budget", 10000)),
                    risk_level=RiskLevel(data.get("risk_level", "Medium")),
                    investment_horizon_years=int(data.get("investment_horizon_years", 10)),
                    monthly_contribution=float(data.get("monthly_contribution", 0)),
                    existing_portfolio=holdings,
                )

                await websocket.send_json({"event": "progress", "stage": "portfolio_agent", "message": "Generating portfolio recommendation..."})
                await websocket.send_json({"event": "progress", "stage": "risk_agent", "message": "Analysing portfolio risk..."})
                await websocket.send_json({"event": "progress", "stage": "market_agent", "message": "Monitoring market conditions..."})

                output = await _orchestrator.run(
                    user_input=data.get("message", "Provide investment recommendation"),
                    user_profile=profile,
                    session_id=session_id,
                )

                await websocket.send_json({
                    "event": "complete",
                    "session_id": session_id,
                    "message": output.final_narrative,
                    "recommendation": output.recommendation.model_dump() if output.recommendation else None,
                    "risk_metrics": output.risk_metrics.model_dump() if output.risk_metrics else None,
                    "market_insight": output.market_insight.model_dump() if output.market_insight else None,
                    "sources": output.sources,
                })

            except Exception as exc:
                logger.error("ws.pipeline_error", error=str(exc))
                await websocket.send_json({"event": "error", "message": str(exc)})

    except WebSocketDisconnect:
        logger.info("ws.disconnected", session_id=session_id)
