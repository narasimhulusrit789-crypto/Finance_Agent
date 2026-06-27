"""
Risk Analysis Agent
────────────────────────────────────────────────────────────────
Evaluates portfolio risk across multiple dimensions:
  • Volatility & drawdown metrics
  • Diversification scoring
  • Concentration risk warnings
  • Risk-adjusted return (Sharpe ratio)
  • Mitigation recommendations
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from agents.models import AgentResponse, AgentRole, RiskLevel, RiskMetrics

logger = structlog.get_logger(__name__)

# Volatility profiles by risk level (annualised std dev %)
_VOL_PROFILES = {
    "Low": {"volatility": 6.5, "beta": 0.45, "max_drawdown": 8.0, "sharpe": 1.1},
    "Medium": {"volatility": 13.0, "beta": 0.85, "max_drawdown": 18.0, "sharpe": 0.85},
    "High": {"volatility": 24.0, "beta": 1.35, "max_drawdown": 35.0, "sharpe": 0.7},
}


def _compute_risk_score(vol: float, beta: float, max_dd: float, diversification: float) -> float:
    """Composite risk score 0–10 (10 = highest risk)."""
    vol_score = min(vol / 3.0, 10.0)
    beta_score = min(beta * 4.0, 10.0)
    dd_score = min(max_dd / 4.0, 10.0)
    div_penalty = (10.0 - diversification)
    score = (vol_score * 0.35 + beta_score * 0.25 + dd_score * 0.25 + div_penalty * 0.15)
    return round(min(max(score, 0.0), 10.0), 2)


def _diversification_score(allocations: list[dict]) -> float:
    """
    Compute a 0–10 diversification score.
    Higher = better diversified.
    Penalises heavy concentration in a single asset class.
    """
    if not allocations:
        return 5.0
    n = len(allocations)
    max_pct = max(a.get("allocation_pct", 0) for a in allocations)
    hhi = sum((a.get("allocation_pct", 0) / 100) ** 2 for a in allocations)  # Herfindahl index
    # Perfect diversification → hhi = 1/n; maximum concentration → hhi = 1
    score = (1 - hhi) / (1 - 1 / max(n, 2)) * 10
    # Penalise if any single class > 70%
    if max_pct > 70:
        score *= 0.7
    elif max_pct > 55:
        score *= 0.85
    return round(min(max(score, 0.0), 10.0), 2)


class RiskAnalysisAgent(BaseAgent):
    """
    Specialist agent for portfolio risk evaluation and mitigation.
    """

    ROLE = AgentRole.RISK_ANALYSIS
    SYSTEM_PROMPT = """You are a quantitative risk analyst and portfolio risk manager with expertise in:
- Modern Portfolio Theory (MPT) and risk-return optimisation
- Value at Risk (VaR), Conditional VaR, and stress testing
- Diversification theory and correlation analysis
- Regulatory risk frameworks (Basel III, UCITS)
- ESG risk factors and geopolitical risk assessment

Your role is to:
1. Evaluate portfolio risk using quantitative metrics (beta, volatility, Sharpe ratio, max drawdown).
2. Identify concentration risks, correlation risks, and tail risks.
3. Assess diversification quality across asset classes, geographies, and sectors.
4. Generate actionable risk mitigation recommendations.
5. Provide clear risk scores with plain-language explanations.

Always explain risk metrics in accessible language while maintaining technical accuracy.
Clearly label high-risk areas and provide specific, actionable mitigation steps."""

    async def execute(self, task: str, payload: dict[str, Any]) -> AgentResponse:
        profile = payload.get("user_profile", {})
        recommendation = payload.get("recommendation", {})
        allocations = recommendation.get("allocations", [])
        risk = profile.get("risk_level", "Medium")

        # ── Step 1: Quantitative risk metrics ─────────────────────
        vol_profile = _VOL_PROFILES.get(risk, _VOL_PROFILES["Medium"])
        div_score = _diversification_score(allocations)
        risk_score = _compute_risk_score(
            vol=vol_profile["volatility"],
            beta=vol_profile["beta"],
            max_dd=vol_profile["max_drawdown"],
            diversification=div_score,
        )

        # Risk label from score
        if risk_score <= 3.5:
            risk_label = RiskLevel.LOW
        elif risk_score <= 6.5:
            risk_label = RiskLevel.MEDIUM
        else:
            risk_label = RiskLevel.HIGH

        # ── Step 2: Concentration warnings ────────────────────────
        warnings = []
        for a in allocations:
            pct = a.get("allocation_pct", 0)
            name = a.get("asset_class", "Unknown")
            if pct > 60:
                warnings.append(f"⚠️  Heavy concentration in {name} ({pct}%) — exceeds 60% threshold.")
            elif pct > 45:
                warnings.append(f"📌 Notable concentration in {name} ({pct}%) — monitor closely.")

        # ── Step 3: LLM-powered detailed risk narrative ───────────
        alloc_str = "\n".join(
            f"  - {a.get('asset_class', 'N/A')}: {a.get('allocation_pct', 0)}%"
            for a in allocations
        )

        prompt = f"""Perform a comprehensive risk analysis for this investment portfolio:

CLIENT PROFILE:
- Risk Tolerance: {risk}
- Investment Goal: {profile.get('investment_goal', 'N/A')}
- Investment Horizon: {profile.get('investment_horizon_years', 10)} years
- Budget: ${float(profile.get('budget', 0)):,.0f}

PORTFOLIO ALLOCATION:
{alloc_str if alloc_str else "No specific allocation provided."}

QUANTITATIVE RISK METRICS:
- Portfolio Beta: {vol_profile['beta']}
- Annualised Volatility: {vol_profile['volatility']}%
- Maximum Drawdown (estimate): {vol_profile['max_drawdown']}%
- Sharpe Ratio (estimate): {vol_profile['sharpe']}
- Diversification Score: {div_score}/10
- Overall Risk Score: {risk_score}/10 ({risk_label.value} Risk)

{f"CONCENTRATION WARNINGS:{chr(10)}" + chr(10).join(warnings) if warnings else "No major concentration issues identified."}

Provide a detailed risk analysis including:
1. Risk profile assessment and what the metrics mean for this investor
2. Top 3 risk factors specific to this portfolio
3. Scenario analysis: How would this portfolio perform in a market downturn (-20%), flat market, and bull market (+25%)?
4. Correlation risks between asset classes in current market conditions
5. Specific mitigation strategies (5–7 actionable recommendations)
6. Portfolio stress test summary
7. Overall risk verdict and investor suitability assessment"""

        risk_narrative = await self._generate(prompt, max_tokens=2000)

        # ── Step 4: Extract structured mitigations ─────────────────
        extract_prompt = f"""From this risk analysis, extract mitigations as JSON.
Analysis preview: {risk_narrative[:1000]}
Return: {{"mitigation_suggestions": ["suggestion1", ...], "key_risks": ["risk1", ...], "stress_scenarios": {{"bear": "...", "base": "...", "bull": "..."}}}}"""

        try:
            raw = await self._generate(extract_prompt, max_tokens=600, temperature=0.2)
            clean = raw.strip().strip("```json").strip("```").strip()
            structured = json.loads(clean)
        except Exception:  # noqa: BLE001
            structured = {
                "mitigation_suggestions": [
                    "Diversify across uncorrelated asset classes",
                    "Set stop-loss limits at 15% portfolio drawdown",
                    "Review and rebalance quarterly",
                    "Maintain 3–6 months of expenses in liquid cash reserves",
                    "Consider tail-risk hedging with put options or inverse ETFs",
                ],
                "key_risks": warnings[:3] if warnings else ["Market risk", "Liquidity risk", "Inflation risk"],
                "stress_scenarios": {
                    "bear": f"Estimated {vol_profile['max_drawdown']}% decline in severe downturn",
                    "base": f"Expected {_calculate_base(risk)}% annual return in normal conditions",
                    "bull": f"Potential {_calculate_bull(risk)}% gain in strong bull market",
                },
            }

        metrics = RiskMetrics(
            portfolio_beta=vol_profile["beta"],
            volatility_pct=vol_profile["volatility"],
            max_drawdown_pct=vol_profile["max_drawdown"],
            sharpe_ratio=vol_profile["sharpe"],
            risk_score=risk_score,
            risk_label=risk_label,
            diversification_score=div_score,
            concentration_warnings=warnings,
            mitigation_suggestions=structured.get("mitigation_suggestions", []),
        )

        return self._wrap_response(
            task=task,
            result={
                "metrics": metrics.model_dump(),
                "narrative": risk_narrative,
                "key_risks": structured.get("key_risks", []),
                "stress_scenarios": structured.get("stress_scenarios", {}),
            },
            reasoning=f"Risk score {risk_score}/10 ({risk_label.value}); diversification {div_score}/10.",
            confidence=0.90,
        )


def _calculate_base(risk: str) -> float:
    return {"Low": 5.0, "Medium": 9.0, "High": 14.0}.get(risk, 9.0)


def _calculate_bull(risk: str) -> float:
    return {"Low": 10.0, "Medium": 18.0, "High": 30.0}.get(risk, 18.0)
