"""
Base agent class — all four specialist agents inherit from this.
"""

from __future__ import annotations

import abc
import time
from typing import Any

import structlog

from agents.models import AgentResponse, AgentRole
from agents.watsonx_client import GraniteClient

logger = structlog.get_logger(__name__)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all finance agents.

    Subclasses must implement:
        - ``ROLE`` class attribute (AgentRole)
        - ``SYSTEM_PROMPT`` class attribute (str)
        - ``execute(task, payload)`` coroutine
    """

    ROLE: AgentRole
    SYSTEM_PROMPT: str = "You are a helpful financial AI assistant."

    def __init__(self, llm: GraniteClient) -> None:
        self._llm = llm
        self._log = logger.bind(agent=self.ROLE.value)

    # ─────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────

    @abc.abstractmethod
    async def execute(self, task: str, payload: dict[str, Any]) -> AgentResponse:
        """Run the agent for a given task and return a structured response."""

    # ─────────────────────────────────────────
    #  Shared helpers
    # ─────────────────────────────────────────

    async def _generate(
        self,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> str:
        """Generate text with the agent's system prompt pre-applied."""
        t0 = time.perf_counter()
        result = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._log.debug(
            "agent.generate",
            task_preview=user_prompt[:80],
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
        return result

    def _wrap_response(
        self,
        task: str,
        result: dict[str, Any],
        reasoning: str = "",
        sources: list[str] | None = None,
        confidence: float = 0.85,
    ) -> AgentResponse:
        return AgentResponse(
            agent=self.ROLE,
            task=task,
            result=result,
            reasoning=reasoning,
            sources=sources or [],
            confidence=confidence,
        )
