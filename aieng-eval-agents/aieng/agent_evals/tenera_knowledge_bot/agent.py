"""Agent factory for the Tenera Knowledge Bot."""

from __future__ import annotations

from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.retrieval import create_retriever
from aieng.agent_evals.retrieval.tools import make_retrieval_tools
from aieng.agent_evals.retrieval.types import AnswerCitation
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig

from .prompts import KNOWLEDGE_BOT_PROMPT


class TeneraKnowledgeBotResponse(BaseModel):
    """Structured final response emitted by the Tenera Knowledge Bot."""

    answer: str = Field(description="The best answer to the user's question.")
    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description="Supporting citations drawn from retrieved chunks.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Missing data, ambiguity, or assumptions that limit confidence.",
    )


def create_tenera_knowledge_bot_agent(
    name: str = "TeneraKnowledgeBot",
    *,
    instructions: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    retrieval_config: TeneraRetrievalConfig | None = None,
    timeout_sec: int | None = None,
) -> LlmAgent:
    """Create a retrieval-backed knowledge bot for the building-code corpus.

    This intentionally keeps ``LlmAgent`` as the one non-parity exception to
    ``knowledge_qa`` because Tenera relies on ADK structured output schema
    enforcement for its JSON response.
    """
    config = Configs()  # type: ignore[call-arg]
    resolved_config = retrieval_config or config.tenera_retrieval
    retriever, chunk_store = create_retriever(resolved_config)
    tools = make_retrieval_tools(retriever, chunk_store)

    return LlmAgent(
        name=name,
        model=model or config.default_planner_model,
        instruction=instructions or KNOWLEDGE_BOT_PROMPT,
        tools=tools,
        generate_content_config=GenerateContentConfig(
            http_options=HttpOptions(timeout=timeout_sec * 1000) if timeout_sec is not None else None,
            temperature=config.default_temperature if temperature is None else temperature,
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
        output_schema=TeneraKnowledgeBotResponse,
    )
