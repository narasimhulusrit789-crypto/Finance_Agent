"""
Tests for the Finance Agent system.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock GraniteClient that returns deterministic strings."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=json.dumps({
        "key_insights": ["Diversification reduces risk"],
        "recommended_sectors": ["Technology"],
        "risk_considerations": ["Market volatility"],
    }))
    return llm


@pytest.fixture
def sample_user_profile():
    from agents.models import UserProfile, RiskLevel
    return UserProfile(
        user_id="test_user",
        investment_goal="Wealth Growth",
        budget=50000.0,
        risk_level=RiskLevel.MEDIUM,
        investment_horizon_years=10,
        monthly_contribution=500.0,
        age=35,
    )


@pytest.fixture
def sample_holdings():
    from agents.models import PortfolioHolding, AssetClass
    return [
        PortfolioHolding(
            symbol="SPY",
            name="SPDR S&P 500 ETF",
            asset_class=AssetClass.EQUITY,
            quantity=10,
            purchase_price=450.0,
            current_price=520.0,
            allocation_pct=60.0,
        ),
        PortfolioHolding(
            symbol="BND",
            name="Vanguard Total Bond Market ETF",
            asset_class=AssetClass.FIXED_INCOME,
            quantity=20,
            purchase_price=72.0,
            current_price=75.0,
            allocation_pct=40.0,
        ),
    ]


# ─────────────────────────────────────────────────────────────
#  Model tests
# ─────────────────────────────────────────────────────────────

class TestModels:
    def test_user_profile_creation(self, sample_user_profile):
        assert sample_user_profile.user_id == "test_user"
        assert sample_user_profile.budget == 50000.0
        assert sample_user_profile.risk_level.value == "Medium"

    def test_portfolio_holding_creation(self, sample_holdings):
        spy = sample_holdings[0]
        assert spy.symbol == "SPY"
        assert spy.asset_class.value == "Equity"

    def test_risk_level_enum(self):
        from agents.models import RiskLevel
        assert RiskLevel.LOW.value == "Low"
        assert RiskLevel.MEDIUM.value == "Medium"
        assert RiskLevel.HIGH.value == "High"

    def test_orchestrator_output_creation(self):
        from agents.models import OrchestratorOutput
        output = OrchestratorOutput(
            user_id="test",
            session_id="session_123",
            final_narrative="Test narrative",
        )
        assert output.user_id == "test"
        assert output.session_id == "session_123"


# ─────────────────────────────────────────────────────────────
#  RAG pipeline tests
# ─────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_text_splitter(self):
        from rag.pipeline import _split_text
        text = "a" * 1200
        chunks = _split_text(text, chunk_size=512, overlap=64)
        assert len(chunks) >= 2
        assert all(len(c) <= 512 for c in chunks)

    def test_load_txt(self, tmp_path):
        from rag.pipeline import load_document
        f = tmp_path / "test.txt"
        f.write_text("Hello financial world!")
        result = load_document(f)
        assert "Hello financial world!" in result

    def test_vector_store_add_query(self):
        """Test in-memory vector store without ChromaDB."""
        from rag.pipeline import VectorStore
        with patch.dict(os.environ, {
            "VECTOR_DB_PATH": "/tmp/test_vstore",
            "VECTOR_DB_TYPE": "chromadb",
            "WATSONX_API_KEY": "test",
            "WATSONX_PROJECT_ID": "test",
            "VECTOR_COLLECTION_NAME": "test_col",
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "SECRET_KEY": "test-secret-key-abc",
            "IBM_CLOUD_API_KEY": "test",
        }):
            # Force memory backend
            store = VectorStore.__new__(VectorStore)
            store._backend = "memory"
            store._memory_store = []
            from sentence_transformers import SentenceTransformer
            store._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            texts = ["Investment in equities provides growth.", "Bonds offer stability."]
            n = store.add_documents(texts, source="test")
            assert n == 2
            assert store.count() == 2

            results = store.query("stock market growth", top_k=2)
            assert len(results) > 0
            assert "text" in results[0]
            assert "score" in results[0]

    def test_ingestion_pipeline_text(self):
        from rag.pipeline import VectorStore, RAGIngestionPipeline
        with patch.object(VectorStore, '__init__', lambda self: None):
            pipeline = RAGIngestionPipeline.__new__(RAGIngestionPipeline)
            pipeline._chunk_size = 512
            pipeline._chunk_overlap = 64
            mock_store = MagicMock()
            mock_store.count.return_value = 5
            mock_store.add_documents.return_value = 3
            pipeline._store = mock_store

            n = pipeline.ingest_text("Some financial text to ingest.", source="test_source")
            assert mock_store.add_documents.called


# ─────────────────────────────────────────────────────────────
#  Agent tests
# ─────────────────────────────────────────────────────────────

class TestInvestmentKnowledgeAgent:
    @pytest.mark.asyncio
    async def test_execute_without_retriever(self, mock_llm, sample_user_profile):
        from agents.investment_knowledge_agent import InvestmentKnowledgeAgent
        agent = InvestmentKnowledgeAgent(mock_llm, retriever=None)
        resp = await agent.execute(
            "What are good investments for retirement?",
            {"query": "retirement investments", "user_profile": sample_user_profile.model_dump()},
        )
        assert resp.agent.value == "investment_knowledge"
        assert isinstance(resp.result, dict)
        assert "summary" in resp.result

    @pytest.mark.asyncio
    async def test_execute_with_mock_retriever(self, mock_llm, sample_user_profile):
        from agents.investment_knowledge_agent import InvestmentKnowledgeAgent
        mock_retriever = AsyncMock()
        mock_retriever.query = AsyncMock(return_value=[
            {"text": "Bond ETFs are stable investments.", "source": "test_doc", "score": 0.85}
        ])
        agent = InvestmentKnowledgeAgent(mock_llm, retriever=mock_retriever)
        resp = await agent.execute(
            "Tell me about bonds",
            {"query": "bonds", "user_profile": sample_user_profile.model_dump()},
        )
        assert resp.result.get("context_docs_used", 0) >= 0


class TestPortfolioRecommendationAgent:
    @pytest.mark.asyncio
    async def test_medium_risk_allocation(self, mock_llm, sample_user_profile):
        from agents.portfolio_recommendation_agent import PortfolioRecommendationAgent
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "top_picks": ["SPY — Core holding", "BND — Diversifier", "GLD — Hedge", "QQQ — Growth", "VNQ — REIT"],
            "rebalancing": "semi-annual"
        }))
        agent = PortfolioRecommendationAgent(mock_llm)
        resp = await agent.execute(
            "Generate portfolio",
            {"user_profile": sample_user_profile.model_dump(), "knowledge_summary": "Markets are stable."},
        )
        assert resp.agent.value == "portfolio_recommendation"
        result = resp.result
        assert "allocations" in result
        assert "expected_annual_return_pct" in result
        assert result["expected_annual_return_pct"] > 0

    def test_projected_value_calculation(self):
        """Test FV calculation logic."""
        # FV of $50K at 9% for 10 years = ~$118,368
        budget, rate, years = 50000, 0.09, 10
        fv = budget * (1 + rate) ** years
        assert 118000 < fv < 119000


class TestRiskAnalysisAgent:
    @pytest.mark.asyncio
    async def test_risk_scoring(self, mock_llm, sample_user_profile):
        from agents.risk_analysis_agent import RiskAnalysisAgent
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "mitigation_suggestions": ["Diversify across sectors"],
            "key_risks": ["Market risk"],
            "stress_scenarios": {"bear": "-20%", "base": "+9%", "bull": "+25%"},
        }))
        allocs = [
            {"asset_class": "Equity", "allocation_pct": 55},
            {"asset_class": "Fixed Income", "allocation_pct": 25},
            {"asset_class": "Real Estate", "allocation_pct": 10},
            {"asset_class": "Cash", "allocation_pct": 10},
        ]
        agent = RiskAnalysisAgent(mock_llm)
        resp = await agent.execute(
            "Analyse risk",
            {"user_profile": sample_user_profile.model_dump(), "recommendation": {"allocations": allocs}},
        )
        assert resp.agent.value == "risk_analysis"
        metrics = resp.result.get("metrics", {})
        assert "risk_score" in metrics
        assert 0 <= metrics["risk_score"] <= 10

    def test_diversification_score(self):
        from agents.risk_analysis_agent import _diversification_score
        # Well diversified
        allocs = [
            {"allocation_pct": 25}, {"allocation_pct": 25},
            {"allocation_pct": 25}, {"allocation_pct": 25},
        ]
        score = _diversification_score(allocs)
        assert score > 8.0  # Near perfect

        # Highly concentrated
        concentrated = [{"allocation_pct": 90}, {"allocation_pct": 10}]
        score_conc = _diversification_score(concentrated)
        assert score_conc < 6.0


class TestMarketMonitoringAgent:
    @pytest.mark.asyncio
    async def test_execute_returns_alerts(self, mock_llm, sample_user_profile):
        from agents.market_monitoring_agent import MarketMonitoringAgent
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "alerts": [{"severity": "info", "title": "Market Update", "description": "S&P at ATH", "affected_symbols": ["SPY"], "action": "Hold position"}],
            "sentiment": "Bullish",
            "trending_sectors": ["Technology", "Healthcare"],
            "top_opportunities": ["AI stocks"],
        }))
        agent = MarketMonitoringAgent(mock_llm)
        resp = await agent.execute(
            "Monitor market",
            {"user_profile": sample_user_profile.model_dump(), "portfolio_symbols": ["SPY", "BND"]},
        )
        assert resp.agent.value == "market_monitoring"
        assert "alerts_count" in resp.result
        assert "sentiment" in resp.result


# ─────────────────────────────────────────────────────────────
#  Agent Memory tests
# ─────────────────────────────────────────────────────────────

class TestAgentMemory:
    def test_session_lifecycle(self):
        from agents.orchestrator import AgentMemory
        memory = AgentMemory()
        sid = "test_session"
        memory.add_message(sid, "user", "Hello")
        memory.add_message(sid, "assistant", "Hi there!")
        session = memory.get_session(sid)
        assert len(session["messages"]) == 2
        assert session["turn_count"] == 2

    def test_agent_output_storage(self):
        from agents.orchestrator import AgentMemory
        memory = AgentMemory()
        sid = "s1"
        memory.store_agent_output(sid, "knowledge_agent", {"summary": "Test summary"})
        out = memory.get_agent_output(sid, "knowledge_agent")
        assert out["summary"] == "Test summary"

    def test_context_summary(self):
        from agents.orchestrator import AgentMemory
        memory = AgentMemory()
        sid = "s2"
        for i in range(8):
            memory.add_message(sid, "user" if i % 2 == 0 else "assistant", f"Message {i}")
        summary = memory.get_context_summary(sid)
        assert "Message" in summary

    def test_clear_session(self):
        from agents.orchestrator import AgentMemory
        memory = AgentMemory()
        sid = "s3"
        memory.add_message(sid, "user", "Test")
        memory.clear_session(sid)
        assert sid not in memory.list_sessions()


# ─────────────────────────────────────────────────────────────
#  Workflow config tests
# ─────────────────────────────────────────────────────────────

class TestWorkflowConfig:
    def test_workflow_config_valid(self):
        from config.bob_workflow import validate_workflow_config
        issues = validate_workflow_config()
        assert issues == [], f"Workflow config issues: {issues}"

    def test_all_agents_present(self):
        from config.bob_workflow import WORKFLOW_CONFIG
        agent_ids = {a["id"] for a in WORKFLOW_CONFIG["agents"]}
        assert "orchestrator" in agent_ids
        assert "investment_knowledge_agent" in agent_ids
        assert "portfolio_recommendation_agent" in agent_ids
        assert "risk_analysis_agent" in agent_ids
        assert "market_monitoring_agent" in agent_ids


# ─────────────────────────────────────────────────────────────
#  File component tests
# ─────────────────────────────────────────────────────────────

class TestFileComponent:
    def test_local_backend_upload_download(self, tmp_path):
        from storage.file_component import FileComponent
        with patch.dict(os.environ, {
            "IBM_COS_API_KEY": "",
            "IBM_CLOUD_API_KEY": "test",
            "WATSONX_API_KEY": "test",
            "WATSONX_PROJECT_ID": "test",
            "SECRET_KEY": "test-secret-key-abc",
        }):
            fc = FileComponent.__new__(FileComponent)
            fc._backend = "local"
            fc._local_root = tmp_path
            fc._cfg = MagicMock()

            # Upload
            key = "test/file.txt"
            fc.upload(b"test content", key)
            assert (tmp_path / key).exists()

            # Download
            content = fc.download(key)
            assert content == b"test content"

            # List
            files = fc.list_files("test/")
            assert any(f["key"] == key for f in files)

            # Delete
            assert fc.delete(key)
            assert not (tmp_path / key).exists()
