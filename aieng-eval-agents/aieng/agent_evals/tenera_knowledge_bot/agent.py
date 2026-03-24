"""Agent factory for the Tenera Knowledge Bot."""

from pydantic import BaseModel, Field

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.langfuse import init_tracing
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig

from .prompts import KNOWLEDGE_BOT_PROMPT


class TeneraKnowledgeEvidence(BaseModel):
    """One evidence item supporting the final answer."""

    statement: str = Field(description="A concise fact supported by the queried database results.")
    source: str = Field(description="The table, view, or query result the fact came from.")


class TeneraKnowledgeBotResponse(BaseModel):
    """Structured final response emitted by the Tenera Knowledge Bot."""

    answer: str = Field(description="The best answer to the user's question.")
    evidence: list[TeneraKnowledgeEvidence] = Field(
        default_factory=list,
        description="Evidence items that support the answer.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Missing data, ambiguity, or assumptions that limit confidence.",
    )


def create_tenera_knowledge_bot_agent(
    name: str = "TeneraKnowledgeBot",
    *,
    instructions: str | None = None,
    timeout_sec: int | None = None,
    enable_tracing: bool = True,
) -> LlmAgent:
    """Create a DB-backed knowledge bot that answers questions from structured data."""
    client_manager = AsyncClientManager.get_instance()
    db = DbManager.get_instance().tenera_knowledge_bot_db(agent_name=name)

    if enable_tracing:
        init_tracing(service_name=name)

    return LlmAgent(
        name=name,
        model=client_manager.configs.default_planner_model,
        instruction=instructions or KNOWLEDGE_BOT_PROMPT,
        tools=[db.get_schema_info, db.execute],
        generate_content_config=GenerateContentConfig(
            http_options=HttpOptions(timeout=timeout_sec * 1000) if timeout_sec is not None else None,
            temperature=0.0,
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
        output_schema=TeneraKnowledgeBotResponse,
    )
