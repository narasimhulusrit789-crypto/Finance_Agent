"""
Market Monitoring Agent
────────────────────────────────────────────────────────────────
Monitors financial news, stock market trends, and generates
real-time alerts for investment opportunities and risks.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from agents.models import AgentResponse, AgentRole, MarketAlert, MarketInsight

logger = structlog.get_logger(__name__)


class MarketMonitoringAgent(BaseAgent):
    """
    Tracks live market conditions and fires contextual alerts.
    """

    ROLE = AgentRole.MARKET_MONITORING
    SYSTEM_PROMPT = """You are a market intelligence analyst and financial news analyst specialising in:
- Real-time equity and bond market monitoring
- Macroeconomic indicator analysis (GDP, CPI, Fed rates, employment)
- Sector rotation analysis and emerging market trends
- Corporate earnings analysis and guidance interpretation
- Geopolitical risk and its market impact assessment
- Technical analysis (moving averages, RSI, support/resistance)

Your role is to:
1. Monitor market conditions and identify significant trends.
2. Generate timely, actionable alerts for investment opportunities and risks.
3. Provide sentiment analysis based on news flow and market data.
4. Identify sector rotation opportunities and emerging themes.
5. Synthesise complex market data into clear, actionable insights.

Be concise but comprehensive. Distinguish between signal and noise.
Rate alert severity accurately: info (FYI), warning (monitor closely), critical (immediate action needed)."""

    def __init__(self, llm, market_data_client=None) -> None:
        super().__init__(llm)
        self._market_data = market_data_client  # Optional live data feed

    async def execute(self, task: str, payload: dict[str, Any]) -> AgentResponse:
        profile = payload.get("user_profile", {})
        portfolio_symbols = payload.get("portfolio_symbols", [])
        recommendation = payload.get("recommendation", {})

        # ── Step 1: Fetch market data (live or simulated) ─────────
        market_data = await self._fetch_market_data(portfolio_symbols)

        # ── Step 2: Generate market intelligence report ───────────
        portfolio_context = (
            f"Monitoring portfolio: {', '.join(portfolio_symbols[:10])}"
            if portfolio_symbols
            else "General market monitoring."
        )

        recommended_instruments = []
        for alloc in recommendation.get("allocations", [])[:5]:
            recommended_instruments.extend(alloc.get("suggested_instruments", [])[:2])

        prompt = f"""Generate a comprehensive market intelligence report and investment alerts.

CLIENT CONTEXT:
- Investment Goal: {profile.get('investment_goal', 'General Investment')}
- Risk Tolerance: {profile.get('risk_level', 'Medium')}
- Investment Horizon: {profile.get('investment_horizon_years', 10)} years
- {portfolio_context}

RECOMMENDED INSTRUMENTS TO MONITOR:
{chr(10).join(f"  - {inst}" for inst in recommended_instruments[:10]) if recommended_instruments else "  Broad market ETFs"}

CURRENT MARKET SNAPSHOT:
{json.dumps(market_data, indent=2)}

Provide a detailed market intelligence report covering:
1. Overall market sentiment and direction (Bullish/Bearish/Neutral) with reasoning
2. Top 3 investment opportunities identified in current conditions
3. Top 3 risks to monitor in the next 30-90 days
4. Sector performance overview — which sectors are leading/lagging
5. Key economic indicators to watch (Fed meeting, earnings, data releases)
6. Specific alerts for the instruments in this portfolio
7. Short-term (1–4 weeks) and medium-term (3–6 months) market outlook

Format alerts clearly with severity levels: INFO / WARNING / CRITICAL"""

        market_report = await self._generate(prompt, max_tokens=2000)

        # ── Step 3: Generate structured alerts ────────────────────
        alert_prompt = f"""Based on this market report, generate 3–5 specific investment alerts as JSON.
Report preview: {market_report[:1200]}
User risk: {profile.get('risk_level', 'Medium')}

Return JSON:
{{
  "alerts": [
    {{"severity": "info|warning|critical", "title": "...", "description": "...", "affected_symbols": ["..."], "action": "..."}},
    ...
  ],
  "sentiment": "Bullish|Bearish|Neutral",
  "trending_sectors": ["sector1", "sector2", "sector3"],
  "top_opportunities": ["opp1", "opp2", "opp3"]
}}"""

        try:
            raw = await self._generate(alert_prompt, max_tokens=800, temperature=0.3)
            clean = raw.strip().strip("```json").strip("```").strip()
            alert_data = json.loads(clean)
        except Exception:  # noqa: BLE001
            alert_data = self._default_alerts(profile.get("risk_level", "Medium"))

        # ── Step 4: Build typed alert objects ────────────────────
        alerts = [
            MarketAlert(
                alert_id=str(uuid.uuid4())[:8],
                severity=a.get("severity", "info"),
                title=a.get("title", "Market Update"),
                description=a.get("description", ""),
                affected_symbols=a.get("affected_symbols", []),
                action_suggestion=a.get("action", ""),
            )
            for a in alert_data.get("alerts", [])
        ]

        insight = MarketInsight(
            summary=market_report,
            trending_sectors=alert_data.get("trending_sectors", []),
            top_gainers=market_data.get("top_gainers", []),
            top_losers=market_data.get("top_losers", []),
            sentiment=alert_data.get("sentiment", "Neutral"),
            alerts=alerts,
        )

        return self._wrap_response(
            task=task,
            result={
                "insight": insight.model_dump(),
                "alerts_count": len(alerts),
                "critical_alerts": sum(1 for a in alerts if a.severity == "critical"),
                "sentiment": insight.sentiment,
                "top_opportunities": alert_data.get("top_opportunities", []),
                "market_data_snapshot": market_data,
            },
            reasoning=f"Generated {len(alerts)} market alerts; sentiment: {insight.sentiment}.",
            confidence=0.82,
        )

    # ─────────────────────────────────────────
    #  Market data helpers
    # ─────────────────────────────────────────

    async def _fetch_market_data(self, symbols: list[str]) -> dict[str, Any]:
        """
        Fetch market data — uses live feed if configured, else returns
        a representative snapshot for demonstration purposes.
        """
        if self._market_data:
            try:
                return await self._market_data.fetch(symbols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("market_data.fetch_failed", error=str(exc))

        # Simulated market snapshot (replace with live feed in production)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "indices": {
                "S&P 500": {"value": 5234.18, "change_pct": 0.34},
                "NASDAQ": {"value": 16421.36, "change_pct": 0.51},
                "DOW JONES": {"value": 38945.84, "change_pct": 0.18},
                "VIX": {"value": 14.82, "change_pct": -2.1},
                "10Y Treasury Yield": {"value": 4.28, "change_pct": 0.05},
            },
            "sector_performance": {
                "Technology": 1.2,
                "Healthcare": 0.4,
                "Financials": 0.6,
                "Energy": -0.3,
                "Consumer Discretionary": 0.8,
                "Utilities": -0.5,
                "Real Estate": -0.2,
                "Industrials": 0.3,
            },
            "top_gainers": [
                {"symbol": "NVDA", "change_pct": 4.2},
                {"symbol": "AMD", "change_pct": 3.1},
                {"symbol": "TSLA", "change_pct": 2.8},
            ],
            "top_losers": [
                {"symbol": "XOM", "change_pct": -1.8},
                {"symbol": "CVX", "change_pct": -1.5},
                {"symbol": "PFE", "change_pct": -1.2},
            ],
            "macro_indicators": {
                "fed_funds_rate": "5.25–5.50%",
                "cpi_yoy": "3.1%",
                "unemployment": "3.7%",
                "gdp_growth": "2.5%",
            },
            "market_news": [
                "Fed signals potential rate cuts if inflation continues declining",
                "AI sector earnings beat expectations — tech leads market rally",
                "Corporate bond spreads tighten as credit conditions improve",
                "Geopolitical tensions weigh on energy sector outlook",
            ],
        }

    @staticmethod
    def _default_alerts(risk_level: str) -> dict:
        base = {
            "alerts": [
                {
                    "severity": "info",
                    "title": "Market Rally — Technology Sector",
                    "description": "AI and semiconductor stocks driving broad tech sector gains.",
                    "affected_symbols": ["QQQ", "NVDA", "AMD"],
                    "action": "Review tech allocation — consider trimming if overweight above target.",
                },
                {
                    "severity": "warning",
                    "title": "Rising Treasury Yields",
                    "description": "10-year Treasury yield at 4.28% may pressure bond valuations.",
                    "affected_symbols": ["BND", "AGG", "TLT"],
                    "action": "Consider shortening bond duration to reduce interest rate sensitivity.",
                },
                {
                    "severity": "info",
                    "title": "Fed Rate Cut Signal",
                    "description": "Federal Reserve signals potential rate cuts in H2 2024.",
                    "affected_symbols": ["SPY", "GLD", "REIT"],
                    "action": "Rate-sensitive assets (REITs, bonds) may benefit — review allocation.",
                },
            ],
            "sentiment": "Bullish",
            "trending_sectors": ["Technology", "Healthcare", "Financials"],
            "top_opportunities": [
                "AI/ML infrastructure plays (NVDA, AMD, MSFT)",
                "Rate-sensitive REITs ahead of potential rate cuts",
                "Emerging market ETFs as dollar weakens",
            ],
        }
        if risk_level == "High":
            base["alerts"].append({
                "severity": "critical",
                "title": "High Volatility Alert",
                "description": "VIX elevated — increased short-term market volatility expected.",
                "affected_symbols": ["SPY", "QQQ"],
                "action": "Consider protective puts or reducing leverage positions temporarily.",
            })
        return base
