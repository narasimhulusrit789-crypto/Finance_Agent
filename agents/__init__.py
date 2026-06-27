"""Agents package — exports all agent classes and models."""

from agents.base_agent import BaseAgent
from agents.investment_knowledge_agent import InvestmentKnowledgeAgent
from agents.market_monitoring_agent import MarketMonitoringAgent
from agents.models import (
    AgentMessage,
    AgentResponse,
    AgentRole,
    AssetAllocation,
    AssetClass,
    InvestmentRecommendation,
    MarketAlert,
    MarketInsight,
    OrchestratorOutput,
    PortfolioHolding,
    RiskLevel,
    RiskMetrics,
    UserProfile,
)
from agents.orchestrator import AgentOrchestrator, AgentMemory
from agents.portfolio_recommendation_agent import PortfolioRecommendationAgent
from agents.risk_analysis_agent import RiskAnalysisAgent
from agents.watsonx_client import GraniteClient

__all__ = [
    "BaseAgent",
    "GraniteClient",
    "InvestmentKnowledgeAgent",
    "PortfolioRecommendationAgent",
    "RiskAnalysisAgent",
    "MarketMonitoringAgent",
    "AgentOrchestrator",
    "AgentMemory",
    # Models
    "AgentMessage",
    "AgentResponse",
    "AgentRole",
    "AssetAllocation",
    "AssetClass",
    "InvestmentRecommendation",
    "MarketAlert",
    "MarketInsight",
    "OrchestratorOutput",
    "PortfolioHolding",
    "RiskLevel",
    "RiskMetrics",
    "UserProfile",
]
