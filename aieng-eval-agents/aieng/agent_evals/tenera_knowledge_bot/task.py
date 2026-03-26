"""Compatibility task wrapper for the Tenera Knowledge Bot."""

from __future__ import annotations

import json
import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from langfuse.experiment import ExperimentItem

from .bot import TeneraKnowledgeBot


logger = logging.getLogger(__name__)


class TeneraKnowledgeBotTask:
    """Langfuse-compatible task wrapper that creates a fresh bot per item."""

    def __init__(
        self,
        *,
        config: Configs | None = None,
        model: str | None = None,
        retrieval_config: TeneraRetrievalConfig | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._retrieval_config = retrieval_config
        self._timeout_sec = timeout_sec

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:
        """Run the bot on one experiment item and return structured output."""
        del kwargs

        item_input = item.get("input") if isinstance(item, dict) else item.input
        prompt_text = item_input if isinstance(item_input, str) else json.dumps(item_input, ensure_ascii=False, indent=2)

        bot = TeneraKnowledgeBot(
            config=self._config,
            model=self._model,
            retrieval_config=self._retrieval_config,
            timeout_sec=self._timeout_sec,
        )

        try:
            response = await bot.answer_async(prompt_text)
        finally:
            await bot.close()

        if not response.text:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            item_id = metadata.get("id") if metadata else "unknown"
            logger.warning("No final response produced for item=%s", item_id)
            return None

        result = {
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "caveats": response.caveats,
        }

        # Reuse the Langfuse experiment span rather than creating a separate
        # OTEL parent span in the task wrapper. This matches the pattern used
        # by the other evaluation tasks and keeps rich metadata on the item run.
        client_manager = AsyncClientManager.get_instance()
        span_metadata: dict[str, Any] = {
            **result,
            "tool_calls": response.tool_calls,
            "tool_responses": response.tool_responses,
            "reasoning_chain": response.reasoning_chain,
            "total_duration_ms": response.total_duration_ms,
            "token_usage": response.token_usage.model_dump(),
        }
        try:
            span_metadata["agent_trace_id"] = client_manager.langfuse_client.get_current_trace_id()
        except Exception:
            logger.debug("Could not fetch current Langfuse trace ID.", exc_info=True)

        try:
            client_manager.langfuse_client.update_current_span(metadata=span_metadata)
        except Exception:
            logger.debug("Could not update current Langfuse span metadata.", exc_info=True)

        return result

    async def close(self) -> None:
        """Compatibility no-op: task instances no longer own a long-lived bot."""
