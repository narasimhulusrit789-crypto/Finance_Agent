"""
Multi-Agent Orchestrator — IBM BOB Orchestration Layer
────────────────────────────────────────────────────────────────
Coordinates the four specialist agents in a directed pipeline:

  1. Investment Knowledge Agent  → RAG + financial research
  2. Portfolio Recommendation Agent → personalised allocation
  3. Risk Analysis Agent          → risk scoring + mitigations
  4. Market Monitoring Agent      → live alerts + market insights
  5. Granite LLM (final pass)     → synthesise into one narrative

The orchestrator manages:
  • Agent execution order and dependency passing
  • Inter-agent message routing
  • Session-scoped conversational memory
  • Tool calling and error recovery
  • Final response assembly
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any

import structlog

from agents.investment_knowledge_agent import InvestmentKnowledgeAgent
from agents.market_monitoring_agent import MarketMonitoringAgent
from agents.models import (
    AgentMessage,
    AgentResponse,
    AgentRole,
    OrchestratorOutput,
    PortfolioHolding,
    UserProfile,
)
from agents.portfolio_recommendation_agent import PortfolioRecommendationAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.watsonx_client import GraniteClient

logger = structlog.get_logger(__name__)


class AgentMemory:
    """
    In-process session memory for multi-turn conversations.
    Stores conversation history, intermediate agent outputs,
    and accumulated context per user session.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "agent_outputs": {},
                "user_profile": None,
                "created_at": datetime.utcnow().isoformat(),
                "turn_count": 0,
            }
        return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get_session(session_id)
        session["messages"].append(
            {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
        )
        session["turn_count"] += 1

    def store_agent_output(self, session_id: str, agent: str, output: dict) -> None:
        self.get_session(session_id)["agent_outputs"][agent] = output

    def get_agent_output(self, session_id: str, agent: str) -> dict | None:
        return self.get_session(session_id)["agent_outputs"].get(agent)

    def set_user_profile(self, session_id: str, profile: dict) -> None:
        self.get_session(session_id)["user_profile"] = profile

    def get_context_summary(self, session_id: str) -> str:
        """Return a condensed context string for LLM prompts."""
        session = self.get_session(session_id)
        msgs = session["messages"][-6:]  # last 6 turns
        lines = [f"{m['role'].upper()}: {m['content'][:200]}" for m in msgs]
        return "\n".join(lines)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class AgentOrchestrator:
    """
    IBM BOB multi-agent orchestrator.

    Execution flow (configurable):
    ┌───────────────────────────────────────────────┐
    │  User Input                                   │
    │     ↓                                         │
    │  Knowledge Agent  ──context──▶ Portfolio Agent│
    │                                     ↓         │
    │                              Risk Agent        │
    │                                     ↓         │
    │                           Market Monitor Agent │
    │                                     ↓         │
    │                          Synthesis (Granite)   │
    │                                     ↓         │
    │                          OrchestratorOutput   │
    └───────────────────────────────────────────────┘
    """

    ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert financial advisor AI orchestrator. 
Your role is to synthesise outputs from four specialist AI agents into a coherent, 
actionable investment recommendation for the user.

You have access to:
1. Financial knowledge research (from documents and market data)
2. Personalised portfolio recommendations with asset allocations
3. Quantitative risk analysis with mitigation strategies
4. Live market monitoring with current alerts and opportunities

Synthesise these into a clear, professional, and actionable final recommendation.
Be specific, avoid vague generalisations, and always tie recommendations to the user's stated goals.
End with a concise executive summary and top 3 immediate action items."""

    def __init__(self, retriever=None) -> None:
        self._llm = GraniteClient()
        self._memory = AgentMemory()
        self._message_log: list[AgentMessage] = []

        # Initialise specialist agents
        self._knowledge_agent = InvestmentKnowledgeAgent(self._llm, retriever=retriever)
        self._portfolio_agent = PortfolioRecommendationAgent(self._llm)
        self._risk_agent = RiskAnalysisAgent(self._llm)
        self._market_agent = MarketMonitoringAgent(self._llm)

        logger.info("AgentOrchestrator.initialised")

    # ─────────────────────────────────────────
    #  Primary entry point
    # ─────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        user_profile: UserProfile,
        session_id: str | None = None,
    ) -> OrchestratorOutput:
        """
        Execute the full multi-agent pipeline for a user query.

        Parameters
        ----------
        user_input:  The user's natural-language query / chat message.
        user_profile: Structured user investment profile.
        session_id:  Optional session ID for memory continuity.

        Returns
        -------
        OrchestratorOutput with all agent results consolidated.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        log = logger.bind(session_id=session_id, user_id=user_profile.user_id)
        log.info("orchestrator.run.start")

        # Persist profile and user turn
        profile_dict = user_profile.model_dump()
        self._memory.set_user_profile(session_id, profile_dict)
        self._memory.add_message(session_id, "user", user_input)

        portfolio_symbols = [h.symbol for h in user_profile.existing_portfolio]

        # ── Stage 1: Knowledge retrieval (can run standalone) ─────
        log.info("stage.1.knowledge_agent")
        knowledge_resp = await self._dispatch(
            agent=self._knowledge_agent,
            task="Retrieve financial knowledge relevant to user's investment query",
            payload={
                "query": user_input,
                "user_profile": profile_dict,
            },
            session_id=session_id,
        )
        knowledge_summary = knowledge_resp.result.get("summary", "")

        # ── Stage 2 & 3: Portfolio + Risk run concurrently ────────
        log.info("stage.2-3.portfolio_and_risk_agents")
        portfolio_task = self._dispatch(
            agent=self._portfolio_agent,
            task="Generate personalised portfolio recommendation",
            payload={
                "user_profile": profile_dict,
                "knowledge_summary": knowledge_summary,
            },
            session_id=session_id,
        )
        # Risk needs portfolio output — run sequentially
        portfolio_resp = await portfolio_task
        recommendation_dict = portfolio_resp.result

        risk_resp = await self._dispatch(
            agent=self._risk_agent,
            task="Evaluate portfolio risk and generate mitigation strategies",
            payload={
                "user_profile": profile_dict,
                "recommendation": recommendation_dict,
            },
            session_id=session_id,
        )

        # ── Stage 4: Market monitoring (parallel with risk) ───────
        log.info("stage.4.market_monitoring_agent")
        market_resp = await self._dispatch(
            agent=self._market_agent,
            task="Monitor market conditions and generate investment alerts",
            payload={
                "user_profile": profile_dict,
                "portfolio_symbols": portfolio_symbols,
                "recommendation": recommendation_dict,
            },
            session_id=session_id,
        )

        # ── Stage 5: Synthesis ────────────────────────────────────
        log.info("stage.5.synthesis")
        final_narrative = await self._synthesise(
            user_input=user_input,
            profile=profile_dict,
            knowledge=knowledge_resp.result,
            portfolio=portfolio_resp.result,
            risk=risk_resp.result,
            market=market_resp.result,
            session_id=session_id,
        )

        self._memory.add_message(session_id, "assistant", final_narrative[:500])

        # ── Assemble output ───────────────────────────────────────
        from agents.models import InvestmentRecommendation, RiskMetrics, MarketInsight
        try:
            rec = InvestmentRecommendation(**recommendation_dict)
        except Exception:
            rec = None
        try:
            risk_metrics = RiskMetrics(**risk_resp.result.get("metrics", {}))
        except Exception:
            risk_metrics = None
        try:
            market_insight = MarketInsight(**market_resp.result.get("insight", {}))
        except Exception:
            market_insight = None

        all_sources = list(set(
            knowledge_resp.sources
            + portfolio_resp.sources
            + risk_resp.sources
            + market_resp.sources
        ))

        output = OrchestratorOutput(
            user_id=user_profile.user_id,
            session_id=session_id,
            knowledge_summary=knowledge_summary,
            recommendation=rec,
            risk_metrics=risk_metrics,
            market_insight=market_insight,
            final_narrative=final_narrative,
            sources=all_sources,
        )

        elapsed = round(time.perf_counter() - t0, 2)
        log.info("orchestrator.run.complete", elapsed_s=elapsed, sources=len(all_sources))
        return output

    # ─────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────

    async def _dispatch(
        self,
        agent,
        task: str,
        payload: dict[str, Any],
        session_id: str,
    ) -> AgentResponse:
        """Send a task to an agent, log the message, store result."""
        msg = AgentMessage(
            message_id=str(uuid.uuid4())[:8],
            sender=AgentRole.ORCHESTRATOR,
            recipient=agent.ROLE,
            task=task,
            payload=payload,
        )
        self._message_log.append(msg)
        logger.debug("dispatch", to=agent.ROLE.value, task=task[:60])

        try:
            resp = await agent.execute(task, payload)
            self._memory.store_agent_output(session_id, agent.ROLE.value, resp.result)
            return resp
        except Exception as exc:
            logger.error("agent.execute.failed", agent=agent.ROLE.value, error=str(exc))
            # Return a minimal error response so the pipeline continues
            return AgentResponse(
                agent=agent.ROLE,
                task=task,
                result={"error": str(exc), "partial": True},
                reasoning="Agent execution failed — partial result.",
                confidence=0.0,
            )

    async def _synthesise(
        self,
        user_input: str,
        profile: dict,
        knowledge: dict,
        portfolio: dict,
        risk: dict,
        market: dict,
        session_id: str,
    ) -> str:
        """Ask Granite to produce a coherent final recommendation narrative."""
        context_history = self._memory.get_context_summary(session_id)

        # Build compact summaries from each agent
        knowledge_summary = knowledge.get("summary", "")[:600]
        portfolio_summary = portfolio.get("summary", "")[:600]
        risk_narrative = risk.get("narrative", "")[:600]
        market_summary = market.get("insight", {}).get("summary", "")[:400]
        sentiment = market.get("sentiment", "Neutral")
        risk_score = risk.get("metrics", {}).get("risk_score", 0)
        expected_return = portfolio.get("expected_annual_return_pct", 0)
        projected_value = portfolio.get("projected_value_at_horizon", 0)

        prompt = f"""You are synthesising outputs from four specialist investment AI agents into a final recommendation.

USER QUERY: {user_input}

CLIENT PROFILE:
- Goal: {profile.get('investment_goal', 'N/A')}
- Budget: ${float(profile.get('budget', 0)):,.0f}
- Risk: {profile.get('risk_level', 'N/A')}
- Horizon: {profile.get('investment_horizon_years', 0)} years

AGENT OUTPUTS SUMMARY:
━━━ KNOWLEDGE AGENT ━━━
{knowledge_summary}

━━━ PORTFOLIO AGENT ━━━
Expected Return: {expected_return:.1f}% p.a. | Projected Value: ${projected_value:,.0f}
{portfolio_summary}

━━━ RISK AGENT ━━━
Risk Score: {risk_score}/10 | Key Risks Identified
{risk_narrative}

━━━ MARKET AGENT ━━━
Market Sentiment: {sentiment}
{market_summary}

CONVERSATION HISTORY:
{context_history}

Please synthesise all of the above into a comprehensive, personalised investment recommendation covering:

## 📊 Executive Summary
Brief overview of the recommended strategy.

## 💼 Portfolio Allocation
Specific allocation percentages and instruments.

## 📈 Expected Returns
Projected value and return timeline.

## ⚠️ Risk Assessment
Key risks and mitigation strategies.

## 🌍 Market Context
Current market conditions and how they affect this portfolio.

## ✅ Immediate Action Items
Top 3 specific steps to take now.

## 📝 Important Disclaimers
Standard investment disclaimers.

Write in a professional, confident, and accessible tone. Be specific and actionable."""

        return await self._llm.generate(
            prompt=prompt,
            system_prompt=self.ORCHESTRATOR_SYSTEM_PROMPT,
            max_tokens=2500,
        )

    # ─────────────────────────────────────────
    #  Public utilities
    # ─────────────────────────────────────────

    def get_session_history(self, session_id: str) -> list[dict]:
        return self._memory.get_session(session_id)["messages"]

    def get_message_log(self) -> list[dict]:
        return [m.model_dump() for m in self._message_log[-50:]]

    def clear_session(self, session_id: str) -> None:
        self._memory.clear_session(session_id)

    @property
    def active_sessions(self) -> list[str]:
        return self._memory.list_sessions()
