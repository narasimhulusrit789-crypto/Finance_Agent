"""
Shared data models used across all agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AssetClass(str, Enum):
    EQUITY = "Equity"
    FIXED_INCOME = "Fixed Income"
    REAL_ESTATE = "Real Estate"
    COMMODITIES = "Commodities"
    CASH = "Cash"
    CRYPTO = "Cryptocurrency"
    INTERNATIONAL = "International"
    ALTERNATIVE = "Alternative"


class AgentRole(str, Enum):
    INVESTMENT_KNOWLEDGE = "investment_knowledge"
    PORTFOLIO_RECOMMENDATION = "portfolio_recommendation"
    RISK_ANALYSIS = "risk_analysis"
    MARKET_MONITORING = "market_monitoring"
    ORCHESTRATOR = "orchestrator"


# ─────────────────────────────────────────────────────────────
#  User & Portfolio models
# ─────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Captures a user's investment profile from the chat interface."""

    user_id: str
    investment_goal: str = Field(description="e.g. Retirement, Child Education, Wealth Growth")
    budget: float = Field(gt=0, description="Total investable amount in USD")
    risk_level: RiskLevel
    investment_horizon_years: int = Field(gt=0, le=50)
    existing_portfolio: list[PortfolioHolding] = Field(default_factory=list)
    monthly_contribution: float = Field(default=0.0)
    age: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioHolding(BaseModel):
    """A single asset in a user's portfolio."""

    symbol: str
    name: str
    asset_class: AssetClass
    quantity: float
    purchase_price: float
    current_price: float = 0.0
    allocation_pct: float = 0.0


# ─────────────────────────────────────────────────────────────
#  Agent message / task models
# ─────────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """Inter-agent communication envelope."""

    message_id: str
    sender: AgentRole
    recipient: AgentRole
    task: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Standardised agent output."""

    agent: AgentRole
    task: str
    result: dict[str, Any]
    reasoning: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
#  Investment recommendation models
# ─────────────────────────────────────────────────────────────

class AssetAllocation(BaseModel):
    asset_class: AssetClass
    allocation_pct: float
    rationale: str
    suggested_instruments: list[str] = Field(default_factory=list)


class InvestmentRecommendation(BaseModel):
    """Output from the Portfolio Recommendation Agent."""

    user_id: str
    summary: str
    allocations: list[AssetAllocation]
    expected_annual_return_pct: float
    projected_value_at_horizon: float
    rebalancing_frequency: str
    top_picks: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
#  Risk analysis models
# ─────────────────────────────────────────────────────────────

class RiskMetrics(BaseModel):
    """Output from the Risk Analysis Agent."""

    portfolio_beta: float = 1.0
    volatility_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    risk_score: float = Field(0.0, ge=0.0, le=10.0)
    risk_label: RiskLevel = RiskLevel.MEDIUM
    diversification_score: float = Field(0.0, ge=0.0, le=10.0)
    concentration_warnings: list[str] = Field(default_factory=list)
    mitigation_suggestions: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  Market monitoring models
# ─────────────────────────────────────────────────────────────

class MarketAlert(BaseModel):
    """A market event alert from the Market Monitoring Agent."""

    alert_id: str
    severity: Literal["info", "warning", "critical"]
    title: str
    description: str
    affected_symbols: list[str] = Field(default_factory=list)
    action_suggestion: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MarketInsight(BaseModel):
    """Aggregated market insights."""

    summary: str
    trending_sectors: list[str] = Field(default_factory=list)
    top_gainers: list[dict[str, Any]] = Field(default_factory=list)
    top_losers: list[dict[str, Any]] = Field(default_factory=list)
    sentiment: Literal["Bullish", "Bearish", "Neutral"] = "Neutral"
    alerts: list[MarketAlert] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
#  Orchestrator output
# ─────────────────────────────────────────────────────────────

class OrchestratorOutput(BaseModel):
    """Final aggregated response sent to the UI."""

    user_id: str
    session_id: str
    knowledge_summary: str = ""
    recommendation: Optional[InvestmentRecommendation] = None
    risk_metrics: Optional[RiskMetrics] = None
    market_insight: Optional[MarketInsight] = None
    final_narrative: str = ""
    sources: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


