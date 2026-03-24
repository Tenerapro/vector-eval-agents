# Tenera Knowledge Bot

This is a scaffold for a database-backed knowledge bot that fits the same repo conventions as the other examples.

It is designed for a bot that:

- answers natural-language questions using a relational database,
- uses a read-only SQL tool for retrieval,
- emits structured JSON answers,
- and is evaluated through Langfuse experiments.

## Files

- `implementations/tenera_knowledge_bot/evaluate.py`: run offline Langfuse evaluations
- `implementations/tenera_knowledge_bot/data/langfuse_upload.py`: upload a local JSONL dataset to Langfuse
- `implementations/tenera_knowledge_bot/data/tenera_knowledge_eval.jsonl`: starter evaluation dataset
- `implementations/tenera_knowledge_bot/rubrics/answer_quality.md`: starter LLM-as-a-judge rubric
- `aieng-eval-agents/aieng/agent_evals/tenera_knowledge_bot/`: reusable package code

## Environment

Add the following to `.env`:

```bash
GOOGLE_API_KEY="..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."

TENERA_KNOWLEDGE_BOT_DB__DRIVER="sqlite"
TENERA_KNOWLEDGE_BOT_DB__DATABASE="implementations/tenera_knowledge_bot/data/tenera_knowledge.db"
TENERA_KNOWLEDGE_BOT_DB__QUERY__MODE="ro"
```

## Evaluation Flow

1. Put your bot's evaluation set into `implementations/tenera_knowledge_bot/data/tenera_knowledge_eval.jsonl`
2. Upload it to Langfuse:

```bash
uv run --env-file .env python -m implementations.tenera_knowledge_bot.data.langfuse_upload
```

3. Run the evaluation:

```bash
uv run --env-file .env python -m implementations.tenera_knowledge_bot.evaluate
```

## Dataset Format

Use JSONL records with this shape:

```json
{"input":"How many active customers do we have?","expected_output":{"answer":"There are 124 active customers.","evidence":["Query over the customers table filtered to active = true returned 124 rows."]},"metadata":{"id":"customers-active-count"}}
```

The starter evaluator uses an LLM judge, so the exact wording can vary as long as the answer is materially correct and grounded.
