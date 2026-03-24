"""Langfuse task wrapper for the Tenera Knowledge Bot."""

import getpass
import json
import logging
import uuid
from typing import Any

from aieng.agent_evals.db_manager import DbManager
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse.experiment import ExperimentItem

from .agent import TeneraKnowledgeBotResponse, create_tenera_knowledge_bot_agent


logger = logging.getLogger(__name__)


class TeneraKnowledgeBotTask:
    """Langfuse-compatible task wrapper for Tenera Knowledge Bot evaluations."""

    def __init__(self, *, agent: LlmAgent | None = None) -> None:
        self._agent = agent or create_tenera_knowledge_bot_agent()
        self._runner = Runner(
            app_name="tenera_knowledge_bot",
            agent=self._agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:
        """Run the bot on one experiment item and return structured output."""
        del kwargs

        item_input = item.get("input") if isinstance(item, dict) else item.input
        prompt_text = item_input if isinstance(item_input, str) else json.dumps(item_input, ensure_ascii=False, indent=2)
        message = types.Content(parts=[types.Part(text=prompt_text)], role="user")

        final_text: str | None = None
        async for event in self._runner.run_async(
            session_id=str(uuid.uuid4()),
            user_id=getpass.getuser(),
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts if part.text)

        if not final_text:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            item_id = metadata.get("id") if metadata else "unknown"
            logger.warning("No final response produced for item=%s", item_id)
            return None

        try:
            return TeneraKnowledgeBotResponse.model_validate_json(final_text.strip()).model_dump()
        except Exception:
            return TeneraKnowledgeBotResponse.model_validate(json.loads(final_text)).model_dump()

    async def close(self) -> None:
        """Close runner and DB resources."""
        await self._runner.close()
        DbManager.get_instance().tenera_knowledge_bot_db().close()
