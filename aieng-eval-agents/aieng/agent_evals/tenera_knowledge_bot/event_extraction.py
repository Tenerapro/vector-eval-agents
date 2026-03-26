"""Event extraction utilities for ADK agent events."""

from __future__ import annotations

import logging
from typing import Any

# Use the wrapper logger so tool-call capture can hook a single stable logger.
logger = logging.getLogger("aieng.agent_evals.tenera_knowledge_bot.bot")


def extract_tool_calls(event: Any) -> list[dict[str, Any]]:
    """Extract tool calls from an ADK runner event.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[dict[str, Any]]
        List of tool call dicts with ``name`` and ``args`` keys.
    """
    if not hasattr(event, "get_function_calls"):
        return []
    function_calls = event.get_function_calls()
    if not function_calls:
        return []

    tool_calls = []
    for fc in function_calls:
        tool_call_info = {
            "name": getattr(fc, "name", "unknown"),
            "args": getattr(fc, "args", {}),
        }
        tool_calls.append(tool_call_info)
        logger.info("Tool call: %s(%s)", tool_call_info["name"], tool_call_info["args"])
    return tool_calls


def extract_tool_responses(event: Any) -> list[dict[str, Any]]:
    """Extract tool responses from an ADK runner event.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[dict[str, Any]]
        List of tool response dicts with ``name`` and ``response`` keys.
    """
    if not hasattr(event, "get_function_responses"):
        return []
    function_responses = event.get_function_responses()
    if not function_responses:
        return []

    responses = []
    for fr in function_responses:
        name = getattr(fr, "name", None) or getattr(fr, "id", "unknown")
        response_data = getattr(fr, "response", {})

        if isinstance(response_data, dict):
            error = response_data.get("error") or response_data.get("status") == "error"
            if error:
                error_msg = response_data.get("error", "Unknown error")
                logger.warning("Tool error: %s failed - %s", name, error_msg)
            else:
                logger.info("Tool response: %s completed", name)
        else:
            logger.info("Tool response: %s completed", name)

        responses.append({"name": name, "response": response_data})
    return responses


def extract_final_response(event: Any) -> str | None:
    """Extract final response text from an event, filtering out thought parts."""
    if not hasattr(event, "is_final_response") or not event.is_final_response():
        return None
    if not hasattr(event, "content") or not event.content:
        return None
    if not hasattr(event.content, "parts") or not event.content.parts:
        return None

    response_parts = []
    for part in event.content.parts:
        if getattr(part, "thought", False):
            continue
        if hasattr(part, "text") and part.text:
            response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else None


def extract_thoughts(event: Any) -> str:
    """Extract thinking/reasoning content from event parts."""
    if not hasattr(event, "content") or not event.content:
        return ""
    if not hasattr(event.content, "parts") or not event.content.parts:
        return ""

    thoughts = []
    for part in event.content.parts:
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            thoughts.append(part.text)
    return "\n".join(thoughts)


def extract_event_text(event: Any) -> str:
    """Extract all text content from event parts."""
    if not (
        hasattr(event, "content")
        and event.content
        and hasattr(event.content, "parts")
        and event.content.parts
    ):
        return ""
    parts = [part.text for part in event.content.parts if hasattr(part, "text") and part.text]
    return "\n".join(parts)
