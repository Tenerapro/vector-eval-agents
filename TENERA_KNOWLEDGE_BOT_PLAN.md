# Tenera Knowledge Bot: One-Off Ingestion + Swappable Retrieval Evaluation

## Problem Statement

We need a maintainable way to turn the Florida building code source text into a searchable knowledge base that the Tenera Knowledge Bot can use to answer questions, and we need an evaluation pipeline that remains stable even if the retrieval implementation changes later.

The current gap is twofold:

- there is no ingestion pipeline that transforms the raw source text into a structured, searchable knowledge store
- the current bot/eval scaffold is not yet organized around a retrieval abstraction that can be swapped later for PostgreSQL `tsvector` lexical search and `pgvector` semantic search without rewriting the agent or the evaluation flow

The desired outcome is:

- a one-off ingestion command that prepares the knowledge base before local use or evaluation
- a local hybrid retrieval backend for v1
- a generic evaluation harness that targets the agent contract, not a specific search implementation
- a top-level plan document in the repo that clearly explains the problem, architecture, defaults, and next implementation steps

## Summary

Build a one-off ingestion pipeline that converts the Florida building code text into a local hybrid knowledge base, then refactor the Tenera bot and evaluation flow so retrieval is tool/backend swappable without changing the agent-facing contract.

Defaults chosen:

- Retrieval target in v1: local hybrid search
- Lexical backend in v1: SQLite FTS5
- Vector backend in v1: embeddings from an OpenAI-compatible API
- Eval target: local agent factory
- Standard response contract: `answer + citations`

## Implementation Changes

### Planning artifact

- Keep this document as the top-level design note and source of truth for the implementation.
- Treat the decisions here as binding defaults unless a later revision updates the plan.

### Knowledge base and ingestion

- Add a dedicated ingestion pipeline under `implementations/tenera_knowledge_bot/ingestion/` with three stages: parse, chunk, index.
- Parse `Florida_Building_Codes_Combined_2023.txt` into structured section records using the existing document markers and headings:
  - code family / volume
  - chapter or appendix
  - section label
  - section title
  - source slug
  - raw section text
- Chunk by section first; split oversized sections into ordered subchunks while preserving parent section metadata.
- Write the processed corpus into a local SQLite knowledge-base file, separate from raw assets.
- Create normalized tables for:
  - documents
  - sections
  - chunks
  - chunk embeddings
- Create an FTS5 virtual table over chunk text plus key metadata fields needed for exact retrieval.
- Store enough metadata on each chunk for answer citations and deterministic eval:
  - chunk id
  - section id
  - chapter
  - section label
  - title
  - source file
  - chunk index
- Make ingestion a one-off CLI command that can be rerun safely; default behavior should rebuild the local KB from source text.

### Retrieval abstraction and tooling

- Introduce a backend-swappable retrieval layer under `implementations/tenera_knowledge_bot/retrieval/`.
- Define stable interfaces for:
  - chunk store / corpus access
  - lexical retriever
  - vector retriever
  - hybrid retriever
  - citation/result normalization
- Implement v1 adapters:
  - `SqliteFtsLexicalRetriever`
  - `OpenAIEmbeddingIndexer`
  - `SqliteHybridRetriever` that merges lexical and vector candidates with deterministic rank fusion
- Make the agent-facing tools call only the retrieval service, never SQLite directly.
- Provide stable retrieval tools such as:
  - `search_knowledge_base(query, top_k, chapter_filter=None)`
  - `get_chunk_by_id(chunk_id)`
  - `get_section_context(section_id)`
- Standardize retrieval results to always include:
  - excerpt text
  - citation metadata
  - retrieval score
  - retrieval method tags such as lexical/vector/hybrid
- Keep the retrieval interface generic enough that later PostgreSQL adapters can replace only the backend layer:
  - Postgres lexical via `tsvector`
  - Postgres vector via `pgvector`
  - Postgres hybrid via the same service contract

#### Retrieval protocol design (implementation detail)

The retrieval layer uses Python `Protocol` classes so backends are structurally typed — no inheritance required, and an HTTP client or a Postgres adapter satisfies the contract if it has the right methods.

**Core result model** — a single Pydantic model shared by every backend:

```python
class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkCitation          # section_id, chapter, section_label, title, source_file, chunk_index
    retrieval_method: Literal["lexical", "vector", "hybrid"]
```

**Retrieval protocols** — three narrow protocols that backends implement:

```python
class LexicalRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 10, filters: RetrievalFilters | None = None) -> list[RetrievalResult]: ...

class VectorRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 10, filters: RetrievalFilters | None = None) -> list[RetrievalResult]: ...

class ChunkStore(Protocol):
    def get_chunk(self, chunk_id: str) -> RetrievalResult | None: ...
    def get_section_chunks(self, section_id: str) -> list[RetrievalResult]: ...
```

`RetrievalFilters` is a plain dataclass (`chapter: str | None`, `section_label: str | None`) so filters are backend-agnostic.

**`HybridRetriever`** is a concrete class, not a protocol — it composes a `LexicalRetriever` + `VectorRetriever` and applies rank fusion:

```python
class HybridRetriever:
    def __init__(self, lexical: LexicalRetriever, vector: VectorRetriever, *, fusion: FusionStrategy = RRFFusion()):
        ...
    def search(self, query: str, *, top_k: int = 10, filters: RetrievalFilters | None = None) -> list[RetrievalResult]:
        ...
```

`FusionStrategy` is itself a protocol (`def fuse(self, lexical: list[RetrievalResult], vector: list[RetrievalResult], top_k: int) -> list[RetrievalResult]`) so weighting / reciprocal rank fusion / learned fusion can be swapped independently.

**Why protocols instead of ABCs**: a remote HTTP retrieval service only needs a thin wrapper with the right method signatures — no base class import, no registration, no diamond inheritance. This is critical for the HTTP adapter case where the implementation lives in a different package or is auto-generated from an OpenAPI spec.

#### Backend registry and factory

A single `create_retriever` factory resolves the active backend from config and returns the composed retrieval stack:

```python
# retrieval/__init__.py
_BACKEND_REGISTRY: dict[str, type[RetrieverBackend]] = {}

def register_backend(name: str, cls: type[RetrieverBackend]) -> None:
    _BACKEND_REGISTRY[name] = cls

def create_retriever(settings: TeneraRetrievalConfig) -> tuple[HybridRetriever, ChunkStore]:
    backend_cls = _BACKEND_REGISTRY[settings.backend]        # e.g. "sqlite", "postgres", "http"
    backend = backend_cls.from_config(settings)               # each backend knows how to build itself
    return backend.hybrid_retriever(), backend.chunk_store()
```

`RetrieverBackend` is a protocol with three factory methods:

```python
class RetrieverBackend(Protocol):
    @classmethod
    def from_config(cls, settings: TeneraRetrievalConfig) -> Self: ...
    def lexical_retriever(self) -> LexicalRetriever: ...
    def vector_retriever(self) -> VectorRetriever: ...
    def chunk_store(self) -> ChunkStore: ...
    def hybrid_retriever(self) -> HybridRetriever:
        # default: compose lexical + vector with config-specified fusion
        ...
```

Each backend self-registers at import time:

```python
# retrieval/backends/sqlite_backend.py
class SqliteRetrieverBackend:
    ...
register_backend("sqlite", SqliteRetrieverBackend)

# retrieval/backends/http_backend.py
class HttpRetrieverBackend:
    """Delegates all retrieval to a remote HTTP service.
    Expects endpoints: POST /search, GET /chunks/{id}, GET /sections/{id}/chunks.
    Response bodies must conform to the RetrievalResult schema."""
    ...
register_backend("http", HttpRetrieverBackend)

# retrieval/backends/postgres_backend.py  (future)
class PostgresRetrieverBackend:
    ...
register_backend("postgres", PostgresRetrieverBackend)
```

**Switching backends** requires only a config change — no code changes to the agent, tools, or evaluation:

```bash
# .env
TENERA_RETRIEVAL__BACKEND=sqlite          # v1 default
# TENERA_RETRIEVAL__BACKEND=http          # point to a remote service
# TENERA_RETRIEVAL__BACKEND=postgres      # future
```

#### Agent tool layer

The agent never sees backends. Three tool functions wrap the retrieval stack and are passed to the `LlmAgent` as `tools=[search_knowledge_base, get_chunk_by_id, get_section_context]`:

```python
# retrieval/tools.py
def make_retrieval_tools(retriever: HybridRetriever, store: ChunkStore) -> list[Callable]:
    """Build agent-facing tool functions closed over a concrete retrieval stack."""

    def search_knowledge_base(query: str, top_k: int = 10, chapter_filter: str | None = None) -> str:
        filters = RetrievalFilters(chapter=chapter_filter)
        results = retriever.search(query, top_k=top_k, filters=filters)
        return _format_results_for_agent(results)   # markdown table with citations

    def get_chunk_by_id(chunk_id: str) -> str: ...
    def get_section_context(section_id: str) -> str: ...

    return [search_knowledge_base, get_chunk_by_id, get_section_context]
```

This replaces the current pattern where the agent receives `[db.get_schema_info, db.execute]` and writes raw SQL. The agent now calls domain-specific retrieval tools, and the backend behind them is invisible.

#### HTTP backend detail

For the "simple HTTP request" swap case, the `HttpRetrieverBackend` wraps a base URL and maps each protocol method to an HTTP call:

```python
class HttpLexicalRetriever:
    def __init__(self, base_url: str, api_key: str | None = None): ...

    def search(self, query: str, *, top_k: int = 10, filters: RetrievalFilters | None = None) -> list[RetrievalResult]:
        resp = httpx.post(f"{self.base_url}/search/lexical", json={"query": query, "top_k": top_k, "filters": asdict(filters)})
        resp.raise_for_status()
        return [RetrievalResult.model_validate(r) for r in resp.json()["results"]]
```

Config for HTTP:

```bash
TENERA_RETRIEVAL__BACKEND=http
TENERA_RETRIEVAL__HTTP_BASE_URL=https://my-search-service.example.com
TENERA_RETRIEVAL__HTTP_API_KEY=sk-...
```

No changes to the agent, tools, eval harness, or dataset format.

### Agent and evaluation pipeline

- Refactor the Tenera bot so it depends on a retriever factory from config, not on a hardcoded DB tool implementation.
- Update the bot output schema to standardize on:
  - `answer`
  - `citations`
  - optional `caveats`
- Treat citations as required for successful grounded answers in normal operation.
- Build a generic local-agent evaluation harness that accepts:
  - an agent factory
  - a retrieval tool backend selection
  - a dataset name or dataset builder input
- Convert `fl_building_code_test_set.csv` into the offline eval dataset used by Langfuse:
  - `input` from `question`
  - `expected_output.answer` from `answer`
  - `expected_output.supporting_text` from `ground_truth`
  - metadata including `chapter`
- Keep the evaluator generic so retrieval tools can be swapped without changing dataset format or output contract.
- Split evaluation into two layers:
  - deterministic retrieval evaluation
  - answer-quality evaluation
- Deterministic retrieval evaluation should measure at least:
  - expected chapter hit in top-k
  - grounding text overlap / expected excerpt hit
  - top-1 and top-k retrieval success
- LLM-as-a-judge should run only after the final answer is produced and should score:
  - factual correctness
  - grounding to retrieved citations
  - completeness
  - citation usefulness
- Keep trace evaluation optional and secondary; the primary generic harness should work even if only the local agent contract is swapped.

### Config and public interfaces

- Add Tenera-specific config for:
  - KB path
  - retrieval backend selection
  - embedding model / endpoint selection
  - top-k defaults
  - hybrid weighting or fusion mode
- Preserve a stable public factory shape for future backends:
  - `create_retriever(settings) -> Retriever`
  - `create_tenera_knowledge_bot_agent(...)`
  - `evaluate(agent_factory=..., retriever_backend=...)`
- Do not hardcode SQLite-specific assumptions into the evaluator, rubric, or agent prompt.

#### Config model (implementation detail)

Add a `TeneraRetrievalConfig` model to `configs.py` alongside the existing `DatabaseConfig`:

```python
class TeneraRetrievalConfig(BaseModel):
    backend: Literal["sqlite", "postgres", "http"] = "sqlite"

    # SQLite-specific (v1 default)
    kb_path: str | None = None                          # path to the ingested .db file

    # Postgres-specific (future)
    db: DatabaseConfig | None = None                    # reuse existing DatabaseConfig

    # HTTP-specific
    http_base_url: str | None = None
    http_api_key: SecretStr | None = None

    # Shared across all backends
    embedding_model: str = "@cf/baai/bge-m3"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    top_k: int = 10
    fusion_mode: Literal["rrf", "weighted"] = "rrf"     # reciprocal rank fusion by default
    lexical_weight: float = 0.4                          # only used when fusion_mode=weighted
    vector_weight: float = 0.6
```

Add to the top-level `Configs`:

```python
class Configs(BaseSettings):
    ...
    tenera_retrieval: TeneraRetrievalConfig | None = Field(default=None)
```

Env vars use the existing `__` nested delimiter:

```bash
TENERA_RETRIEVAL__BACKEND=sqlite
TENERA_RETRIEVAL__KB_PATH=implementations/tenera_knowledge_bot/data/tenera_knowledge.db
TENERA_RETRIEVAL__EMBEDDING_MODEL=text-embedding-3-small
TENERA_RETRIEVAL__TOP_K=10
```

#### Updated agent factory

The agent factory changes from DB-tool injection to retrieval-tool injection:

```python
def create_tenera_knowledge_bot_agent(
    name: str = "TeneraKnowledgeBot",
    *,
    retrieval_config: TeneraRetrievalConfig | None = None,
    ...
) -> LlmAgent:
    config = retrieval_config or AsyncClientManager.get_instance().configs.tenera_retrieval
    hybrid_retriever, chunk_store = create_retriever(config)
    tools = make_retrieval_tools(hybrid_retriever, chunk_store)

    return LlmAgent(
        name=name,
        tools=tools,                  # no more [db.get_schema_info, db.execute]
        ...
    )
```

The eval harness passes the same config, so swapping `TENERA_RETRIEVAL__BACKEND=http` in `.env` changes both the agent and the evaluation without code edits.

## File Layout

```
aieng-eval-agents/aieng/agent_evals/
├── tenera_knowledge_bot/
│   ├── agent.py                     # updated factory: injects retrieval tools, not raw SQL
│   ├── task.py                      # unchanged — still wraps the agent for Langfuse
│   ├── prompts.py                   # updated prompt: references search tools, not SQL
│   └── __init__.py
├── retrieval/                        # NEW — backend-agnostic retrieval layer
│   ├── __init__.py                  # create_retriever(), register_backend()
│   ├── types.py                     # RetrievalResult, ChunkCitation, RetrievalFilters, protocols
│   ├── fusion.py                    # FusionStrategy protocol, RRFFusion, WeightedFusion
│   ├── hybrid.py                    # HybridRetriever (composes lexical + vector)
│   ├── tools.py                     # make_retrieval_tools() — agent-facing tool closures
│   └── backends/
│       ├── __init__.py
│       ├── sqlite_backend.py        # SqliteRetrieverBackend (FTS5 + local embeddings)
│       ├── http_backend.py          # HttpRetrieverBackend (delegates to remote service)
│       └── postgres_backend.py      # future: PostgresRetrieverBackend
├── configs.py                       # add TeneraRetrievalConfig
└── tools/
    └── sql_database.py              # untouched — still available for other agents

implementations/tenera_knowledge_bot/
├── ingestion/                       # NEW — one-off KB build pipeline
│   ├── __init__.py
│   ├── cli.py                       # python -m implementations.tenera_knowledge_bot.ingestion
│   ├── parser.py                    # TXT → section records
│   ├── chunker.py                   # sections → sized chunks
│   └── indexer.py                   # chunks → SQLite FTS5 + embeddings table
├── data/
│   ├── Florida_Building_Codes_Combined_2023.txt
│   └── fl_building_code_test_set.csv
├── agent.py
├── evaluate.py
└── rubrics/
    └── answer_quality.md
```

## Test Plan

- Ingestion test: source text is parsed into sections and chunks with stable metadata and no empty indexed chunks.
- FTS test: known section-number and exact-term queries retrieve expected chapters/sections.
- Vector test: paraphrased queries retrieve relevant chunks even when exact wording differs.
- Hybrid test: merged retrieval improves recall on representative CSV examples without breaking exact section lookups.
- Tool contract test: retrieval tool outputs always conform to the normalized citation/result schema.
- Agent test: bot returns `answer + citations` and cites retrieved code chunks, not fabricated references.
- Dataset-builder test: CSV rows are transformed into Langfuse-ready examples correctly.
- Evaluation test: deterministic retrieval metrics and LLM-judge metrics both run against the same generic harness.

## Assumptions

- The one-off ingestion pipeline is manually run before evaluation and before normal local usage.
- The building code TXT file is the canonical source for v1 ingestion.
- The CSV test set is the canonical offline evaluation set for v1.
- OpenAI-compatible embeddings are acceptable for v1 vector indexing.
- PostgreSQL `tsvector` and `pgvector` support will be added later by implementing new retrieval adapters behind the same interfaces, not by rewriting the agent or evaluator.
