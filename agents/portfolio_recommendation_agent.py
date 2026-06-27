"""
Portfolio Recommendation Agent
────────────────────────────────────────────────────────────────
Generates personalised investment recommendations based on:
  • User goals, budget, risk tolerance, and investment horizon
  • Current portfolio holdings
  • Knowledge retrieved by the Investment Knowledge Agent
"""

from __future__ import annotations

import json
import math
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from agents.models import (
    AgentResponse,
    AgentRole,
    AssetAllocation,
    AssetClass,
    InvestmentRecommendation,
    RiskLevel,
)

logger = structlog.get_logger(__name__)

# Default strategic asset allocation templates by risk profile
_ALLOCATION_TEMPLATES: dict[str, list[dict]] = {
    "Low": [
        {"asset_class": AssetClass.FIXED_INCOME, "pct": 50, "instruments": ["US Treasury Bonds", "Investment-Grade Corporate Bonds", "Bond ETFs (BND, AGG)"]},
        {"asset_class": AssetClass.EQUITY, "pct": 25, "instruments": ["Dividend Stocks", "S&P 500 ETF (SPY)", "Blue-chip stocks"]},
        {"asset_class": AssetClass.CASH, "pct": 15, "instruments": ["High-Yield Savings", "Money Market Funds", "CDs"]},
        {"asset_class": AssetClass.REAL_ESTATE, "pct": 7, "instruments": ["REITs (VNQ)", "Real Estate ETFs"]},
        {"asset_class": AssetClass.COMMODITIES, "pct": 3, "instruments": ["Gold ETF (GLD)", "Silver ETF"]},
    ],
    "Medium": [
        {"asset_class": AssetClass.EQUITY, "pct": 55, "instruments": ["S&P 500 ETF (SPY)", "Growth ETF (QQQ)", "Mid-cap ETF (MDY)"]},
        {"asset_class": AssetClass.FIXED_INCOME, "pct": 25, "instruments": ["Bond ETFs (BND)", "Corporate Bond ETFs (LQD)", "Treasury Notes"]},
        {"asset_class": AssetClass.REAL_ESTATE, "pct": 10, "instruments": ["REITs (VNQ)", "Real Estate ETFs (XLRE)"]},
        {"asset_class": AssetClass.COMMODITIES, "pct": 5, "instruments": ["Gold ETF (GLD)", "Commodity ETF (DJP)"]},
        {"asset_class": AssetClass.CASH, "pct": 5, "instruments": ["High-Yield Savings", "T-bills"]},
    ],
    "High": [
        {"asset_class": AssetClass.EQUITY, "pct": 75, "instruments": ["Growth ETF (QQQ)", "Small-Cap ETF (IWM)", "Sector ETFs", "Individual Growth Stocks"]},
        {"asset_class": AssetClass.INTERNATIONAL, "pct": 10, "instruments": ["Emerging Markets (EEM)", "International ETF (VXUS)", "Global ETF"]},
        {"asset_class": AssetClass.ALTERNATIVE, "pct": 8, "instruments": ["Hedge Fund ETFs", "Private Equity ETFs (PSP)"]},
        {"asset_class": AssetClass.REAL_ESTATE, "pct": 5, "instruments": ["REITs", "Real Estate Growth ETFs"]},
        {"asset_class": AssetClass.CRYPTO, "pct": 2, "instruments": ["Bitcoin ETF (IBIT)", "Ethereum ETF"]},
    ],
}

# Expected annual return ranges by risk level
_RETURN_ESTIMATES = {"Low": 0.05, "Medium": 0.09, "High": 0.14}


class PortfolioRecommendationAgent(BaseAgent):
    """
    Generates a personalised investment portfolio allocation.
    """

    ROLE = AgentRole.PORTFOLIO_RECOMMENDATION
    SYSTEM_PROMPT = """You are an expert portfolio manager and certified financial planner (CFP) 
with 20+ years of experience in wealth management and asset allocation.

Your role is to:
1. Analyse user financial goals, budget, risk tolerance, and investment horizon.
2. Generate personalised, diversified portfolio recommendations.
3. Provide specific asset allocation with rationale and suggested instruments.
4. Calculate realistic expected returns and projected portfolio values.
5. Recommend rebalancing strategies and contribution plans.

Always follow modern portfolio theory (MPT) principles and fiduciary standards.
Be specific with instrument suggestions (ETF tickers, fund names) while noting they are for educational purposes.
Consider tax efficiency, liquidity needs, and inflation in your recommendations."""

    async def execute(self, task: str, payload: dict[str, Any]) -> AgentResponse:
        profile = payload.get("user_profile", {})
        knowledge_summary = payload.get("knowledge_summary", "")

        risk = profile.get("risk_level", "Medium")
        budget = float(profile.get("budget", 10000))
        horizon = int(profile.get("investment_horizon_years", 10))
        goal = profile.get("investment_goal", "Wealth Growth")
        monthly_contrib = float(profile.get("monthly_contribution", 0))

        # ── Step 1: Compute quantitative allocation ────────────────
        template = _ALLOCATION_TEMPLATES.get(risk, _ALLOCATION_TEMPLATES["Medium"])
        expected_return = _RETURN_ESTIMATES.get(risk, 0.09)

        # Projected value using FV formula with monthly contributions
        fv_lump = budget * (1 + expected_return) ** horizon
        if monthly_contrib > 0:
            monthly_rate = expected_return / 12
            n_months = horizon * 12
            fv_contrib = monthly_contrib * (((1 + monthly_rate) ** n_months - 1) / monthly_rate)
        else:
            fv_contrib = 0.0
        projected_value = fv_lump + fv_contrib

        allocations = [
            AssetAllocation(
                asset_class=item["asset_class"],
                allocation_pct=item["pct"],
                rationale=f"Aligned with {risk} risk profile and {goal} goal.",
                suggested_instruments=item["instruments"],
            )
            for item in template
        ]

        # ── Step 2: LLM-enhanced narrative ────────────────────────
        alloc_summary = "\n".join(
            f"  - {a.asset_class.value}: {a.allocation_pct}% ({', '.join(a.suggested_instruments[:2])})"
            for a in allocations
        )
        existing = profile.get("existing_portfolio", [])
        existing_str = (
            "\n".join(f"  - {h.get('symbol','?')} ({h.get('asset_class','?')}): ${h.get('purchase_price',0)*h.get('quantity',0):,.0f}"
                      for h in existing[:5])
            if existing else "No existing holdings."
        )

        prompt = f"""Generate a comprehensive investment recommendation for this client:

FINANCIAL PROFILE:
- Goal: {goal}
- Available Budget: ${budget:,.0f}
- Monthly Contribution: ${monthly_contrib:,.0f}
- Risk Tolerance: {risk}
- Investment Horizon: {horizon} years
- Existing Portfolio: {existing_str}

PROPOSED ASSET ALLOCATION:
{alloc_summary}

PROJECTED OUTCOME:
- Expected Annual Return: {expected_return*100:.1f}%
- Projected Value at Horizon: ${projected_value:,.0f}

MARKET CONTEXT FROM KNOWLEDGE AGENT:
{knowledge_summary[:800] if knowledge_summary else "General market conditions apply."}

Please provide:
1. Executive summary of the recommendation strategy
2. Rationale for each asset class allocation
3. Top 5 specific investment picks with brief reasoning
4. Rebalancing schedule recommendation
5. Key milestones and portfolio review triggers
6. Important risks and disclaimers

Write in a professional yet accessible tone."""

        narrative = await self._generate(prompt, max_tokens=2000)

        # ── Step 3: Extract top picks ──────────────────────────────
        picks_prompt = f"""From this portfolio recommendation, list exactly 5 top investment picks as JSON.
Context: {narrative[:1000]}
Return: {{"top_picks": ["TICKER/Name - brief rationale", ...], "rebalancing": "quarterly/semi-annual/annual"}}"""
        try:
            picks_raw = await self._generate(picks_prompt, max_tokens=400, temperature=0.2)
            clean = picks_raw.strip().strip("```json").strip("```").strip()
            picks_data = json.loads(clean)
        except Exception:  # noqa: BLE001
            picks_data = {
                "top_picks": [t["instruments"][0] for t in template[:5]],
                "rebalancing": "quarterly" if risk == "High" else "semi-annual",
            }

        recommendation = InvestmentRecommendation(
            user_id=profile.get("user_id", "anonymous"),
            summary=narrative,
            allocations=allocations,
            expected_annual_return_pct=expected_return * 100,
            projected_value_at_horizon=projected_value,
            rebalancing_frequency=picks_data.get("rebalancing", "semi-annual"),
            top_picks=picks_data.get("top_picks", []),
        )

        return self._wrap_response(
            task=task,
            result=recommendation.model_dump(),
            reasoning=f"Applied {risk} risk template; projected ${projected_value:,.0f} over {horizon} years at {expected_return*100:.1f}% annual return.",
            confidence=0.88,
        )
