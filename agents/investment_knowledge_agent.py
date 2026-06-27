"""
Investment Knowledge Agent
────────────────────────────────────────────────────────────────
Retrieves and summarises financial information from:
  • The vector index (RAG)
  • Uploaded company reports / financial documents
  • Market news feeds
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from agents.models import AgentResponse, AgentRole

logger = structlog.get_logger(__name__)


class InvestmentKnowledgeAgent(BaseAgent):
    """
    Specialist agent for financial knowledge retrieval.

    Responsibilities
    ────────────────
    1. Query the vector index with the user's question / context.
    2. Retrieve and rank relevant document chunks.
    3. Synthesise a grounded, cited answer using Granite.
    """

    ROLE = AgentRole.INVESTMENT_KNOWLEDGE
    SYSTEM_PROMPT = """You are a senior financial research analyst with deep expertise in 
investment strategies, equity markets, fixed income, macroeconomics, and portfolio theory.

Your role is to:
1. Retrieve and synthesise financial knowledge from research documents, company reports, and market data.
2. Provide clear, accurate, and well-sourced summaries of financial concepts and investment opportunities.
3. Always cite the source of information when available.
4. Distinguish clearly between factual data and opinion/analysis.
5. Use professional financial terminology while remaining accessible.

Format your responses with clear headings, bullet points for key findings, and a brief summary.
Never fabricate financial data, statistics, or company information."""

    def __init__(self, llm, retriever=None) -> None:
        super().__init__(llm)
        self._retriever = retriever  # RAGRetriever instance (injected)

    async def execute(self, task: str, payload: dict[str, Any]) -> AgentResponse:
        query = payload.get("query", task)
        user_profile = payload.get("user_profile", {})
        context_docs = []
        sources = []

        # ── Step 1: RAG retrieval ──────────────────────────────────
        if self._retriever:
            try:
                results = await self._retriever.query(query, top_k=5)
                context_docs = [r["text"] for r in results]
                sources = [r.get("source", "Knowledge Base") for r in results]
                logger.debug("rag.retrieved", n_docs=len(context_docs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("rag.retrieval_failed", error=str(exc))

        # ── Step 2: Build enriched prompt ─────────────────────────
        context_block = ""
        if context_docs:
            context_block = "\n\n[RETRIEVED KNOWLEDGE]\n" + "\n---\n".join(
                f"[Source {i+1}: {sources[i]}]\n{doc}"
                for i, doc in enumerate(context_docs)
            )

        user_context = ""
        if user_profile:
            user_context = (
                f"\n\n[USER PROFILE]\n"
                f"Goal: {user_profile.get('investment_goal', 'N/A')}\n"
                f"Budget: ${user_profile.get('budget', 0):,.0f}\n"
                f"Risk Level: {user_profile.get('risk_level', 'N/A')}\n"
                f"Horizon: {user_profile.get('investment_horizon_years', 'N/A')} years"
            )

        prompt = f"""Research Task: {query}
{context_block}
{user_context}

Provide a comprehensive financial knowledge summary covering:
1. Key concepts and market context relevant to this query
2. Relevant financial instruments and investment vehicles
3. Historical performance data and benchmarks (if available in context)
4. Current market conditions and outlook
5. Key considerations for investors matching this profile

Structure your response with clear sections and cite sources where applicable."""

        # ── Step 3: Generate response ──────────────────────────────
        response_text = await self._generate(prompt, max_tokens=2048)

        # ── Step 4: Extract key points ────────────────────────────
        extract_prompt = f"""From this financial research summary, extract a JSON list of key investment insights.
Summary: {response_text[:1500]}

Return JSON: {{"key_insights": ["insight1", "insight2", ...], "recommended_sectors": ["sector1", ...], "risk_considerations": ["risk1", ...]}}"""

        try:
            structured_raw = await self._generate(extract_prompt, max_tokens=512, temperature=0.2)
            # Parse JSON even if wrapped in markdown code fences
            clean = structured_raw.strip().strip("```json").strip("```").strip()
            structured = json.loads(clean)
        except Exception:  # noqa: BLE001
            structured = {
                "key_insights": [response_text[:200]],
                "recommended_sectors": [],
                "risk_considerations": [],
            }

        return self._wrap_response(
            task=task,
            result={
                "summary": response_text,
                "key_insights": structured.get("key_insights", []),
                "recommended_sectors": structured.get("recommended_sectors", []),
                "risk_considerations": structured.get("risk_considerations", []),
                "context_docs_used": len(context_docs),
            },
            reasoning=f"Retrieved {len(context_docs)} context documents; synthesised with Granite.",
            sources=sources,
            confidence=min(0.95, 0.6 + len(context_docs) * 0.07),
        )
