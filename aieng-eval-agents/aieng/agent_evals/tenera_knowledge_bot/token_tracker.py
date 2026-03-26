"""Token usage tracking for Gemini models."""

from __future__ import annotations

import logging
import os
from typing import Any

from google.genai import Client
from pydantic import BaseModel


logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("DEFAULT_WORKER_MODEL", "gemini-2.5-flash")

KNOWN_MODEL_LIMITS: dict[str, int] = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}


class TokenUsage(BaseModel):
    """Token usage statistics."""

    latest_prompt_tokens: int = 0
    latest_cached_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    context_limit: int = 1_000_000

    @property
    def context_used_percent(self) -> float:
        if self.context_limit == 0:
            return 0.0
        return (self.latest_prompt_tokens / self.context_limit) * 100

    @property
    def context_remaining_percent(self) -> float:
        return max(0.0, 100.0 - self.context_used_percent)


class TokenTracker:
    """Tracks token usage across agent interactions."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or DEFAULT_MODEL
        self._usage = TokenUsage()
        self._fetch_model_limits()

    def _fetch_model_limits(self) -> None:
        client = None
        try:
            client = Client()
            model_info = client.models.get(model=self._model)
            if model_info.input_token_limit:
                self._usage.context_limit = model_info.input_token_limit
                logger.debug("Model %s context limit: %d", self._model, self._usage.context_limit)
                return
        except Exception as e:
            logger.warning("Failed to fetch model limits from API: %s", e)
        finally:
            if client is not None:
                client.close()

        if self._model in KNOWN_MODEL_LIMITS:
            self._usage.context_limit = KNOWN_MODEL_LIMITS[self._model]
            logger.info("Using known limit for %s: %d", self._model, self._usage.context_limit)
        else:
            logger.warning("Unknown model %s, using default limit: %d", self._model, self._usage.context_limit)

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    def add_from_event(self, event: Any) -> None:
        """Update token counts from an ADK event's usage_metadata."""
        if not hasattr(event, "usage_metadata") or event.usage_metadata is None:
            return

        metadata = event.usage_metadata
        prompt = getattr(metadata, "prompt_token_count", 0) or 0
        cached = getattr(metadata, "cached_content_token_count", 0) or 0
        completion = getattr(metadata, "candidates_token_count", 0) or 0
        total = getattr(metadata, "total_token_count", 0) or 0

        self._usage.latest_prompt_tokens = prompt
        self._usage.latest_cached_tokens = cached
        self._usage.total_prompt_tokens += prompt
        self._usage.total_completion_tokens += completion
        self._usage.total_tokens += total

        logger.debug(
            "Token update: prompt=%d (cached=%d), context: %.1f%% used",
            prompt,
            cached,
            self._usage.context_used_percent,
        )

    def reset(self) -> None:
        """Reset all token counts (keeps context limit)."""
        context_limit = self._usage.context_limit
        self._usage = TokenUsage(
            latest_prompt_tokens=0,
            latest_cached_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            context_limit=context_limit,
        )
