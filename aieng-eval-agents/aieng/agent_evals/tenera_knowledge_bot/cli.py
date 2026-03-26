"""Rich CLI for the Tenera Knowledge Bot.

Usage
-----
Run from the repository root::

    uv run python -m aieng.agent_evals.tenera_knowledge_bot.cli ask "What is the wind speed requirement?"
    uv run python -m aieng.agent_evals.tenera_knowledge_bot.cli ask --log-trace "What is the wind speed requirement?"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import click
from aieng.agent_evals.configs import Configs, TeneraRetrievalConfig
from aieng.agent_evals.evaluation.trace import flush_traces
from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.logging_config import setup_logging
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------


def _display_banner() -> None:
    console.print(
        Panel(
            "[bold]Tenera Knowledge Bot[/bold]\nRetrieval-backed Florida Building Code assistant",
            border_style="blue",
        )
    )


def _display_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return
    table = Table(title="Tool Calls", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool", style="cyan")
    table.add_column("Arguments")
    for i, tc in enumerate(tool_calls, 1):
        args_str = str(tc.get("args", {}))
        if len(args_str) > 120:
            args_str = args_str[:120] + "…"
        table.add_row(str(i), tc.get("name", "unknown"), args_str)
    console.print(table)


def _display_citations(citations: list[Any]) -> None:
    if not citations:
        return
    table = Table(title="Citations", show_lines=True)
    table.add_column("Chapter", style="cyan")
    table.add_column("Section")
    table.add_column("Title")
    table.add_column("Excerpt", max_width=60)
    for c in citations:
        chapter = getattr(c, "chapter", "") if not isinstance(c, dict) else c.get("chapter", "")
        section = getattr(c, "section_label", "") if not isinstance(c, dict) else c.get("section_label", "")
        title = getattr(c, "title", "") if not isinstance(c, dict) else c.get("title", "")
        excerpt = getattr(c, "excerpt", "") if not isinstance(c, dict) else c.get("excerpt", "")
        if len(excerpt) > 120:
            excerpt = excerpt[:120] + "…"
        table.add_row(chapter, section, title, excerpt)
    console.print(table)


def _display_response(response: Any) -> None:
    """Render a :class:`BotResponse` to the console."""
    console.print()
    _display_tool_calls(response.tool_calls)

    console.print(
        Panel(response.answer or response.text, title="[bold green]Answer[/bold green]", border_style="green")
    )

    _display_citations(response.citations)

    if response.caveats:
        console.print(Panel("\n".join(f"• {c}" for c in response.caveats), title="Caveats", border_style="yellow"))

    # Summary line
    usage = response.token_usage
    console.print(
        f"\n[dim]Duration: {response.total_duration_ms}ms | "
        f"Tool calls: {len(response.tool_calls)} | "
        f"Tokens: {usage.total_tokens} "
        f"(prompt {usage.total_prompt_tokens}, completion {usage.total_completion_tokens})[/dim]"
    )


# ------------------------------------------------------------------
# Tracing helpers
# ------------------------------------------------------------------


def _setup_tracing(log_trace: bool) -> bool:
    if not log_trace:
        return False
    enabled = init_tracing()
    if enabled:
        console.print("[green]Langfuse tracing enabled[/green]\n")
    else:
        console.print("[yellow]Could not initialise Langfuse tracing[/yellow]\n")
    return enabled


def _flush_tracing(tracing_enabled: bool) -> None:
    if not tracing_enabled:
        return
    flush_traces()
    console.print("\n[dim]Traces flushed to Langfuse[/dim]")


# ------------------------------------------------------------------
# Retrieval config helper
# ------------------------------------------------------------------


def _build_retrieval_config(backend: str | None) -> TeneraRetrievalConfig | None:
    configs = Configs()  # type: ignore[call-arg]
    retrieval_config = configs.tenera_retrieval
    if retrieval_config is None and backend is None:
        return None
    resolved = retrieval_config or TeneraRetrievalConfig()
    if backend is None:
        return resolved
    return resolved.model_copy(update={"backend": backend})


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Tenera Knowledge Bot CLI."""
    load_dotenv(verbose=True)
    setup_logging(level=logging.INFO, show_time=True, show_path=False)


@cli.command()
@click.argument("question")
@click.option("--log-trace", is_flag=True, default=False, help="Enable Langfuse tracing for this run.")
@click.option(
    "--retrieval-backend",
    type=click.Choice(["sqlite", "http", "postgres"]),
    default=None,
    help="Override the configured retrieval backend.",
)
@click.option("--timeout", default=180, type=int, help="Timeout in seconds for the Gemini API.")
def ask(question: str, log_trace: bool, retrieval_backend: str | None, timeout: int) -> None:
    """Ask the bot a question."""
    asyncio.run(_cmd_ask(question, log_trace=log_trace, retrieval_backend=retrieval_backend, timeout=timeout))


async def _cmd_ask(
    question: str,
    *,
    log_trace: bool,
    retrieval_backend: str | None,
    timeout: int,
) -> None:
    _display_banner()
    tracing_enabled = _setup_tracing(log_trace)

    console.print(Panel(question, title="[bold blue]Question[/bold blue]", border_style="blue"))

    # Import here to avoid circular imports at module level
    from .bot import TeneraKnowledgeBot  # noqa: PLC0415

    retrieval_config = _build_retrieval_config(retrieval_backend)
    bot = TeneraKnowledgeBot(
        retrieval_config=retrieval_config,
        timeout_sec=timeout,
    )

    try:
        response = await bot.answer_async(question)
        _display_response(response)
    finally:
        await bot.close()
        _flush_tracing(tracing_enabled)


# ------------------------------------------------------------------
# Module entry point
# ------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m aieng.agent_evals.tenera_knowledge_bot.cli``."""
    cli()


if __name__ == "__main__":
    main()
