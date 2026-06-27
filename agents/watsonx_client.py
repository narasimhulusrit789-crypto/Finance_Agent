"""
IBM watsonx.ai Granite client wrapper.
Provides a unified async interface for all agents.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class GraniteClient:
    """
    Thin async wrapper around the IBM watsonx.ai ModelInference SDK.

    All public methods are async-safe: heavy SDK calls are executed in a
    thread-pool executor so they do not block the event loop.
    """

    def __init__(self) -> None:
        cfg = get_settings().watsonx
        credentials = Credentials(url=cfg.url, api_key=cfg.api_key)
        self._client = APIClient(credentials=credentials, project_id=cfg.project_id)
        self._model = ModelInference(
            model_id=cfg.model_id,
            api_client=self._client,
            project_id=cfg.project_id,
            params={
                GenParams.MAX_NEW_TOKENS: cfg.max_tokens,
                GenParams.TEMPERATURE: cfg.temperature,
                GenParams.TOP_P: cfg.top_p,
                GenParams.REPETITION_PENALTY: 1.1,
                GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
            },
        )
        self._cfg = cfg
        logger.info("GraniteClient initialised", model=cfg.model_id)

    # ─────────────────────────────────────────
    #  Core generation
    # ─────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from the Granite model (async)."""
        full_prompt = self._build_prompt(prompt, system_prompt)
        params: dict[str, Any] = {}
        if max_tokens is not None:
            params[GenParams.MAX_NEW_TOKENS] = max_tokens
        if temperature is not None:
            params[GenParams.TEMPERATURE] = temperature

        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_text(
                prompt=full_prompt,
                params=params or None,
            ),
        )
        elapsed = time.perf_counter() - t0
        logger.debug("granite.generate", tokens_requested=max_tokens, elapsed_s=round(elapsed, 3))
        return response.strip()

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: str = "",
    ) -> str:
        """
        Generate text and attempt to return JSON-compatible output.
        Adds a schema hint to the prompt if provided.
        """
        json_prompt = prompt
        if schema_hint:
            json_prompt = f"{prompt}\n\nReturn your answer as valid JSON matching this schema:\n{schema_hint}"
        return await self.generate(json_prompt, system_prompt=system_prompt)

    # ─────────────────────────────────────────
    #  Chat-style multi-turn
    # ─────────────────────────────────────────

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Accept an OpenAI-style message list and convert to Granite prompt format.
        Each message: {"role": "system"|"user"|"assistant", "content": "..."}
        """
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        prompt_parts.append("<|assistant|>")
        return await self.generate("\n".join(prompt_parts))

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _build_prompt(user_prompt: str, system_prompt: str) -> str:
        if system_prompt:
            return (
                f"<|system|>\n{system_prompt}\n"
                f"<|user|>\n{user_prompt}\n"
                "<|assistant|>"
            )
        return f"<|user|>\n{user_prompt}\n<|assistant|>"
