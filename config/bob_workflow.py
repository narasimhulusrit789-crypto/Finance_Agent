"""
IBM BOB Multi-Agent Workflow Configuration
────────────────────────────────────────────────────────────────
Defines the complete multi-agent workflow as structured config.
This is used by IBM BOB to understand agent topology, data flow,
tool bindings, and execution policies.
"""

from __future__ import annotations

# IBM BOB Workflow Specification
WORKFLOW_CONFIG = {
    "version": "1.0",
    "name": "Finance Investment Recommendation Workflow",
    "description": "Multi-agent pipeline for AI-powered investment recommendations",
    "platform": "IBM BOB",
    "foundation_model": {
        "provider": "IBM watsonx.ai",
        "model_id": "ibm/granite-4-8b-instruct",
        "endpoint": "https://us-south.ml.cloud.ibm.com",
        "parameters": {
            "max_new_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
        }
    },
    "agents": [
        {
            "id": "orchestrator",
            "name": "Agent Orchestrator",
            "role": "orchestrator",
            "description": "Coordinates all specialist agents and assembles final response",
            "model": "ibm/granite-4-8b-instruct",
            "tools": ["route_to_agent", "synthesize_outputs", "memory_read", "memory_write"],
            "memory": True,
            "execution_mode": "sequential_with_parallel_stages",
        },
        {
            "id": "investment_knowledge_agent",
            "name": "Investment Knowledge Agent",
            "role": "specialist",
            "description": "Retrieves financial knowledge from vector index and documents",
            "model": "ibm/granite-4-8b-instruct",
            "tools": ["vector_search", "document_retrieve", "news_fetch"],
            "rag_enabled": True,
            "vector_collection": "finance_knowledge",
            "system_prompt": "You are a senior financial research analyst...",
            "execution_stage": 1,
            "parallel_safe": False,
        },
        {
            "id": "portfolio_recommendation_agent",
            "name": "Portfolio Recommendation Agent",
            "role": "specialist",
            "description": "Generates personalised portfolio recommendations",
            "model": "ibm/granite-4-8b-instruct",
            "tools": ["calculate_allocation", "compute_projection", "get_instruments"],
            "rag_enabled": False,
            "receives_from": ["investment_knowledge_agent"],
            "execution_stage": 2,
            "parallel_safe": False,
        },
        {
            "id": "risk_analysis_agent",
            "name": "Risk Analysis Agent",
            "role": "specialist",
            "description": "Evaluates portfolio risk and generates mitigation strategies",
            "model": "ibm/granite-4-8b-instruct",
            "tools": ["compute_risk_metrics", "diversification_score", "stress_test"],
            "rag_enabled": False,
            "receives_from": ["portfolio_recommendation_agent"],
            "execution_stage": 3,
            "parallel_safe": False,
        },
        {
            "id": "market_monitoring_agent",
            "name": "Market Monitoring Agent",
            "role": "specialist",
            "description": "Monitors market conditions and generates alerts",
            "model": "ibm/granite-4-8b-instruct",
            "tools": ["fetch_market_data", "news_api", "sentiment_analysis", "fire_alert"],
            "rag_enabled": False,
            "receives_from": ["portfolio_recommendation_agent"],
            "execution_stage": 3,
            "parallel_safe": True,  # Can run in parallel with risk agent
        },
    ],
    "data_flows": [
        {"from": "user_input", "to": "orchestrator"},
        {"from": "orchestrator", "to": "investment_knowledge_agent", "data": ["query", "user_profile"]},
        {"from": "investment_knowledge_agent", "to": "portfolio_recommendation_agent", "data": ["knowledge_summary", "key_insights"]},
        {"from": "portfolio_recommendation_agent", "to": "risk_analysis_agent", "data": ["allocations", "recommendation"]},
        {"from": "portfolio_recommendation_agent", "to": "market_monitoring_agent", "data": ["portfolio_symbols", "recommendation"]},
        {"from": "risk_analysis_agent", "to": "orchestrator", "data": ["risk_metrics", "narrative"]},
        {"from": "market_monitoring_agent", "to": "orchestrator", "data": ["market_insight", "alerts"]},
        {"from": "orchestrator", "to": "user_output", "data": ["final_narrative", "dashboard_data"]},
    ],
    "tools": {
        "vector_search": {
            "type": "rag",
            "collection": "finance_knowledge",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 5,
        },
        "document_retrieve": {
            "type": "file_storage",
            "backend": "ibm_cos",
            "allowed_types": [".pdf", ".docx", ".txt", ".csv", ".xlsx"],
        },
        "news_fetch": {
            "type": "http",
            "endpoint": "https://newsapi.org/v2/everything",
            "auth_header": "X-Api-Key",
        },
        "fetch_market_data": {
            "type": "http",
            "endpoint": "https://finnhub.io/api/v1",
            "auth_header": "X-Finnhub-Token",
        },
        "compute_risk_metrics": {
            "type": "python_function",
            "module": "agents.risk_analysis_agent",
            "function": "_compute_risk_score",
        },
        "fire_alert": {
            "type": "event",
            "destination": "websocket_broadcast",
            "schema": "MarketAlert",
        },
    },
    "memory": {
        "type": "session_memory",
        "backend": "in_process",
        "max_turns": 20,
        "context_window": 6,
    },
    "rag_pipeline": {
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store": "chromadb",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 5,
        "min_similarity": 0.3,
        "supported_formats": [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".md"],
    },
    "file_component": {
        "type": "ibm_cos",
        "bucket": "finance-agent-files",
        "directories": {
            "portfolios": "portfolios/{user_id}/",
            "documents": "documents/{user_id}/",
            "reports": "reports/{user_id}/",
            "knowledge": "knowledge/",
        },
    },
    "security": {
        "authentication": "jwt",
        "authorisation": "role_based",
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "audit_logging": True,
    },
    "deployment": {
        "platform": "IBM Code Engine",
        "region": "us-south",
        "min_replicas": 1,
        "max_replicas": 5,
        "cpu": "2",
        "memory": "4Gi",
        "health_check": "/health",
        "autoscale_metric": "rps",
        "autoscale_target": 100,
    },
}


def get_workflow_config() -> dict:
    """Return the IBM BOB workflow configuration."""
    return WORKFLOW_CONFIG


def validate_workflow_config() -> list[str]:
    """Validate the workflow configuration. Returns list of issues (empty = valid)."""
    issues = []
    required_agents = {"orchestrator", "investment_knowledge_agent", "portfolio_recommendation_agent", "risk_analysis_agent", "market_monitoring_agent"}
    configured_agents = {a["id"] for a in WORKFLOW_CONFIG["agents"]}
    missing = required_agents - configured_agents
    if missing:
        issues.append(f"Missing required agents: {missing}")
    for agent in WORKFLOW_CONFIG["agents"]:
        if "model" not in agent:
            issues.append(f"Agent {agent['id']} missing model configuration")
    return issues
