# Tenera Knowledge Bot

This is a scaffold for a retrieval-backed knowledge bot that fits the same repo conventions as the other examples.

It is designed for a bot that:

- answers natural-language questions using an indexed knowledge base,
- uses retrieval tools instead of raw SQL,
- emits structured JSON answers,
- and is evaluated through Langfuse experiments.

## Files

- `implementations/tenera_knowledge_bot/evaluate.py`: run offline Langfuse evaluations
- `implementations/tenera_knowledge_bot/ingestion/cli.py`: one-off knowledge-base build command
- `implementations/tenera_knowledge_bot/data/langfuse_upload.py`: upload a local JSONL dataset to Langfuse
- `implementations/tenera_knowledge_bot/data/fl_building_code_test_set.csv`: source evaluation set
- `implementations/tenera_knowledge_bot/rubrics/answer_quality.md`: starter LLM-as-a-judge rubric
- `aieng-eval-agents/aieng/agent_evals/tenera_knowledge_bot/`: reusable package code

## Environment

Add the following to `.env`:

```bash
GOOGLE_API_KEY="..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."

TENERA_RETRIEVAL__BACKEND="sqlite"
TENERA_RETRIEVAL__KB_PATH="implementations/tenera_knowledge_bot/data/tenera_knowledge.db"
TENERA_RETRIEVAL__EMBEDDING_MODEL="gemini-embedding-001"
```

## Ingestion

Build the local knowledge base from the source text:

```bash
uv run --env-file .env python -m implementations.tenera_knowledge_bot.ingestion.cli
```

## Evaluation Flow

1. Upload the CSV-derived evaluation set to Langfuse:

```bash
uv run --env-file .env python -m implementations.tenera_knowledge_bot.data.langfuse_upload
```

2. Run the evaluation:

```bash
uv run --env-file .env python -m implementations.tenera_knowledge_bot.evaluate
```

The evaluator uses the provided CSV test set and converts it to Langfuse dataset items automatically.
