"""Tenera Knowledge Bot wrapper with knowledge_qa-style runtime ownership."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from typing import Any

from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.retrieval.types import AnswerCitation
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .agent import TeneraKnowledgeBotResponse, create_tenera_knowledge_bot_agent
from .event_extraction import extract_final_response, extract_thoughts, extract_tool_calls, extract_tool_responses
from .retry import (
    API_RETRY_INITIAL_WAIT,
    API_RETRY_JITTER,
    API_RETRY_MAX_ATTEMPTS,
    API_RETRY_MAX_WAIT,
    MAX_EMPTY_RESPONSE_RETRIES,
    is_context_overflow_error,
    is_retryable_api_error,
)
from .token_tracker import TokenTracker, TokenUsage


warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*ContextCacheConfig.*")
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*EventsCompactionConfig.*")

logger = logging.getLogger(__name__)


class StepExecution(BaseModel):
    """Record of executing a single retrieval step."""

    step_id: int
    tool_used: str
    input_query: str
    output_summary: str = ""
    sources_found: int = 0
    duration_ms: int = 0
    raw_output: str = ""


class BotResponse(BaseModel):
    """Rich response from the Tenera Knowledge Bot with execution metadata."""

    text: str
    answer: str = ""
    citations: list[AnswerCitation] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    execution_trace: list[StepExecution] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_responses: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)
    total_duration_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class TeneraKnowledgeBot:
    """High-level wrapper around the Tenera Knowledge Bot agent.

    Keeps the same runtime ownership pattern as ``KnowledgeGroundedAgent``:
    configuration, session lifecycle, retries, event extraction, and response
    shaping all live in this wrapper instead of being spread across callers.
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
        enable_planning: bool = True,  # kept for interface parity
        enable_caching: bool = True,
        enable_compaction: bool = True,
        compaction_interval: int = 10,
        retrieval_config: TeneraRetrievalConfig | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        del enable_planning  # Tenera uses structured-output execution rather than planner state.

        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_planner_model
        self.temperature = config.default_temperature
        self.timeout_sec = timeout_sec
        self.retrieval_config = retrieval_config or config.tenera_retrieval
        self._enable_compaction = enable_compaction
        self._compaction_interval = compaction_interval

        self._agent = create_tenera_knowledge_bot_agent(
            model=self.model,
            temperature=self.temperature,
            retrieval_config=self.retrieval_config,
            timeout_sec=timeout_sec,
        )
        self._token_tracker = TokenTracker(model=self.model)

        self._session_service = InMemorySessionService()
        self._app: App | None
        if enable_caching or enable_compaction:
            self._app, self._runner = self._create_app_and_runner(enable_compaction=enable_compaction)
        else:
            self._app = None
            self._runner = Runner(
                app_name="tenera_knowledge_bot",
                agent=self._agent,
                session_service=self._session_service,
            )

        self._sessions: dict[str, str] = {}

    def _create_app_and_runner(self, *, enable_compaction: bool) -> tuple[App, Runner]:
        """Create App and Runner using the same wrapper-owned lifecycle as knowledge_qa."""
        app_kwargs: dict[str, Any] = {
            "name": "tenera_knowledge_bot",
            "root_agent": self._agent,
        }

        if enable_compaction:
            summarizer = LlmEventSummarizer(llm=Gemini(model=self.config.default_worker_model))
            app_kwargs["events_compaction_config"] = EventsCompactionConfig(
                compaction_interval=self._compaction_interval,
                overlap_size=1,
                summarizer=summarizer,
            )

        app = App(**app_kwargs)
        runner = Runner(
            app=app,
            session_service=self._session_service,
        )
        return app, runner

    def reset(self) -> None:
        """Reset agent state for a new question."""
        self._sessions.clear()
        self._session_service = InMemorySessionService()
        self._token_tracker = TokenTracker(model=self.model)

        if self._app is not None:
            self._runner = Runner(
                app=self._app,
                session_service=self._session_service,
            )
        else:
            self._runner = Runner(
                app_name="tenera_knowledge_bot",
                agent=self._agent,
                session_service=self._session_service,
            )
        logger.debug("Agent state reset for new question")

    @property
    def adk_agent(self):
        """Return the underlying ADK agent."""
        return self._agent

    @property
    def token_tracker(self) -> TokenTracker:
        """Return the token tracker used for this bot."""
        return self._token_tracker

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session for the given session ID."""
        if session_id is None:
            session_id = str(int(time.time() * 1_000_000))

        if session_id not in self._sessions:
            session = await self._session_service.create_session(
                app_name="tenera_knowledge_bot",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    async def _run_agent_once_inner(self, question: str, adk_session_id: str) -> dict[str, Any]:
        """Run the agent once and collect results."""
        content = types.Content(role="user", parts=[types.Part(text=question)])
        results: dict[str, Any] = {
            "tool_calls": [],
            "tool_responses": [],
            "reasoning_chain": [],
            "final_response": "",
        }

        event_count = 0
        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            event_count += 1
            self._process_event(event, results)

        logger.debug("Processed %d events. Final response length: %d", event_count, len(results["final_response"]))
        return results

    async def _run_agent_once(self, question: str, adk_session_id: str) -> dict[str, Any]:
        """Run the agent once with retry logic for rate limits and context overflow."""

        @retry(
            retry=retry_if_exception(is_retryable_api_error),
            wait=wait_exponential_jitter(
                initial=API_RETRY_INITIAL_WAIT,
                max=API_RETRY_MAX_WAIT,
                jitter=API_RETRY_JITTER,
            ),
            stop=stop_after_attempt(API_RETRY_MAX_ATTEMPTS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        async def _run_with_retry() -> dict[str, Any]:
            return await self._run_agent_once_inner(question, adk_session_id)

        try:
            return await _run_with_retry()
        except RetryError as e:
            original_error = e.last_attempt.exception()
            logger.error("API retry failed after %d attempts: %s", API_RETRY_MAX_ATTEMPTS, original_error)
            raise RuntimeError(
                f"API request failed after {API_RETRY_MAX_ATTEMPTS} retry attempts. Last error: {original_error}"
            ) from original_error
        except ClientError as e:
            if is_context_overflow_error(e):
                logger.warning("Context overflow detected: %s", e)
                logger.warning("Resetting session and retrying with fresh context...")

                self._session_service = InMemorySessionService()
                self._sessions.clear()
                if self._app is not None:
                    self._runner = Runner(app=self._app, session_service=self._session_service)
                else:
                    self._runner = Runner(
                        app_name="tenera_knowledge_bot",
                        agent=self._agent,
                        session_service=self._session_service,
                    )
                new_session_id = await self._get_or_create_session_async()

                try:
                    return await self._run_agent_once_inner(question, new_session_id)
                except Exception as retry_error:
                    logger.error("Retry with fresh session failed: %s", retry_error)
                    raise RuntimeError(
                        f"Context overflow error. Original error: {e}. "
                        f"Retry with fresh session also failed: {retry_error}"
                    ) from e

            raise

    def _process_event(self, event: Any, results: dict[str, Any]) -> None:
        """Process a single event from the agent run loop."""
        self._token_tracker.add_from_event(event)

        new_tool_calls = extract_tool_calls(event)
        results["tool_calls"].extend(new_tool_calls)

        new_tool_responses = extract_tool_responses(event)
        results["tool_responses"].extend(new_tool_responses)

        thoughts = extract_thoughts(event)
        if thoughts:
            results["reasoning_chain"].append(thoughts[:500])

        final_response = extract_final_response(event)
        if final_response:
            results["final_response"] = final_response

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> BotResponse:
        """Answer a question using the retrieval-backed knowledge bot."""
        start_time = time.time()
        logger.info("Answering question: %s...", question[:100])

        adk_session_id = await self._get_or_create_session_async(session_id)
        results: dict[str, Any] = {}
        current_session_id = adk_session_id

        for attempt in range(MAX_EMPTY_RESPONSE_RETRIES + 1):
            results = await self._run_agent_once(question, current_session_id)

            if results.get("final_response", "").strip():
                break

            if attempt < MAX_EMPTY_RESPONSE_RETRIES:
                logger.warning(
                    "Empty model response (attempt %d/%d), creating fresh session and retrying...",
                    attempt + 1,
                    MAX_EMPTY_RESPONSE_RETRIES + 1,
                )
                fresh_session = await self._session_service.create_session(
                    app_name="tenera_knowledge_bot",
                    user_id="user",
                    state={},
                )
                current_session_id = fresh_session.id
            else:
                tool_call_count = len(results.get("tool_calls", []))
                reasoning_count = len(results.get("reasoning_chain", []))
                logger.error(
                    "Empty model response after %d attempts. Tool calls: %d, Reasoning steps: %d. "
                    "The model may have only produced thinking tokens.",
                    MAX_EMPTY_RESPONSE_RETRIES + 1,
                    tool_call_count,
                    reasoning_count,
                )

        total_duration_ms = int((time.time() - start_time) * 1000)
        return self._build_response(results, total_duration_ms)

    def answer(self, question: str, session_id: str | None = None) -> BotResponse:
        """Answer a question synchronously."""
        logger.info("Answering question (sync): %s...", question[:100])
        return asyncio.run(self.answer_async(question, session_id))

    def _create_execution_trace(
        self,
        tool_calls: list[dict[str, Any]],
        tool_responses: list[dict[str, Any]],
        total_duration_ms: int,
    ) -> list[StepExecution]:
        """Create a normalized execution trace from raw tool calls/responses."""
        response_by_name: dict[str, list[Any]] = {}
        for response in tool_responses:
            response_by_name.setdefault(str(response.get("name", "unknown")), []).append(response.get("response"))

        step_duration_ms = total_duration_ms // max(len(tool_calls), 1)
        execution_trace: list[StepExecution] = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = str(tool_call.get("name", "unknown"))
            tool_args = tool_call.get("args", {})
            matching_responses = response_by_name.get(tool_name, [])
            raw_output = ""
            if matching_responses:
                raw_output = str(matching_responses.pop(0))

            execution_trace.append(
                StepExecution(
                    step_id=i + 1,
                    tool_used=tool_name,
                    input_query=str(tool_args),
                    output_summary=f"Tool call {i + 1}",
                    sources_found=0,
                    duration_ms=step_duration_ms,
                    raw_output=raw_output,
                )
            )
        return execution_trace

    def _build_response(self, results: dict[str, Any], total_duration_ms: int) -> BotResponse:
        """Parse the model's JSON output and build a BotResponse."""
        final_text = results.get("final_response", "")
        answer = ""
        citations: list[AnswerCitation] = []
        caveats: list[str] = []
        tool_calls = results.get("tool_calls", [])
        tool_responses = results.get("tool_responses", [])
        execution_trace = self._create_execution_trace(tool_calls, tool_responses, total_duration_ms)

        if final_text:
            try:
                parsed = TeneraKnowledgeBotResponse.model_validate_json(final_text.strip())
            except Exception:
                try:
                    parsed = TeneraKnowledgeBotResponse.model_validate(json.loads(final_text))
                except Exception:
                    logger.warning("Could not parse structured response, using raw text")
                    parsed = None

            if parsed:
                answer = parsed.answer
                citations = parsed.citations
                caveats = parsed.caveats

        logger.info(
            "Agent completed: %d chars, %d tool calls, %d citations",
            len(final_text),
            len(results.get("tool_calls", [])),
            len(citations),
        )

        return BotResponse(
            text=final_text,
            answer=answer,
            citations=citations,
            caveats=caveats,
            execution_trace=execution_trace,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            reasoning_chain=results.get("reasoning_chain", []),
            total_duration_ms=total_duration_ms,
            token_usage=self._token_tracker.usage,
        )

    async def close(self) -> None:
        """Release runner and session resources."""
        await self._runner.close()
