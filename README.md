# 📡 Research Radar

**An agentic RAG system that keeps up with AI research so you don't have to, running 100% locally.**

AI research moves weekly, and every LLM's knowledge is frozen at its training cutoff. Research Radar
continuously ingests new arXiv papers and answers questions about them with **verified,
cited answers**, plus a **statistical trend radar** that shows which research ideas are rising or fading.

It goes beyond "naive RAG" (embed → retrieve → stuff into prompt) with the techniques production systems use in 2026:

| Capability | How |
|---|---|
| **Hybrid retrieval** | Dense embeddings (`bge-small`) + BM25 sparse vectors in one Qdrant collection, fused with Reciprocal Rank Fusion |
| **Two-stage ranking** | Cross-encoder reranker (`ms-marco-MiniLM`) re-scores the top 30 candidates |
| **Contextual chunking** | Section-aware, sentence-boundary chunks with overlap and paper/section context headers |
| **Agentic control loop** | LLM plans and routes the query, rewrites it into multiple searches, grades evidence, retries on gaps (Corrective RAG), abstains when evidence is missing |
| **Grounded answers** | Inline `[n]` citations, plus a verifier pass that checks every claim against its sources |
| **Trend radar** | Phrase-lift statistics over time windows; the LLM narrates the numbers, it doesn't invent them |
| **Always fresh** | Idempotent, incremental ingestion (`radar refresh`) designed to run on a schedule |
| **Measured, not guessed** | Synthetic test-set generation + ablation of every retrieval stage (hit@k, MRR) + answer faithfulness, see [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) |
| **Observable** | Per-request traces with step-level latency (`data/traces/`) |
| **Agent-ready** | Exposed as an **MCP server**, so Claude, Cursor and other AI assistants can use it as a tool |
| **Private and free** | Local LLM (Qwen3-8B via Ollama), local ONNX embedding/rerank models. No API keys, no data leaves the machine |

## Results

Measured on 400 papers / 2,584 chunks with two synthetic test sets of 40 questions each ([full report](docs/EVAL_RESULTS.md)):

| retrieval mode | name-specific questions (MRR) | plain-language questions (hit@8) |
|---|---|---|
| dense only | 0.62 | 0.40 |
| BM25 only | **0.86** | 0.28 |
| **hybrid + rerank** | 0.82 | **0.53** |

BM25 wins when the user knows the exact name and collapses when they don't; hybrid + rerank is the only setup robust to both.
End to end: **88% of answers fully supported by their sources, 100% cited the correct paper**, median ~20 s on a laptop.

## Architecture

```
 arXiv API ─► PDF parse ─► section-aware chunks + context headers ─► dense + BM25 vectors ─► Qdrant
                                                                                    paper metadata ─► SQLite

 question ─► PLAN/ROUTE ─┬─ search ─► multi-query hybrid retrieval ─► RRF ─► rerank ─► GRADE ─┬─► ANSWER [n] ─► VERIFY
                         │                     ▲                                               │
                         │                     └──────── rewrite query (corrective loop) ◄─────┘
                         ├─ trends ─► time-window phrase statistics ─► LLM brief
                         └─ direct ─► short answer
                                         served via CLI · streaming web UI (SSE) · REST API · MCP
```

## Quick start

```bash
# 1. Local LLM (Ollama) — install from https://ollama.com, then:
ollama serve &              # if not already running
ollama pull qwen3:8b

# 2. Get the code and Python deps (https://docs.astral.sh/uv/ for uv)
git clone https://github.com/mdahmedanjarwala105/research-radar.git && cd research-radar
uv sync

# 3. Build the index (~5–10 min: arXiv rate-limits to 1 request / 3 s)
uv run radar ingest --topic "retrieval augmented generation" --max-papers 400 --fulltext 40 --days 120

# 4. Use it
uv run radar ask "What are the newest techniques to reduce hallucination in RAG?"
uv run radar ask "What's trending in RAG research lately?"
uv run radar search "graph-based retrieval" --mode hybrid_rerank   # inspect raw retrieval
uv run radar trends --weeks 8
uv run radar serve        # web UI at http://127.0.0.1:8000
```

Keep it fresh with a nightly job (`crontab -e`):

```
0 6 * * * cd /path/to/RAG_AI && uv run radar refresh --topic "retrieval augmented generation"
```

### Use it from Claude Code / Claude Desktop (MCP)

```bash
claude mcp add research-radar -- uv --directory /path/to/RAG_AI run radar mcp
```

Tools exposed: `search_papers`, `ask_research_question`, `research_trends`, `get_paper`.

### Evaluate

```bash
uv run radar eval build --n 40 --style specific     # questions naming methods
uv run radar eval build --n 40 --style conceptual   # plain-language questions
uv run radar eval run --answers 8   # retrieval ablation + end-to-end answer scoring
```

> Embedded Qdrant allows one process at a time: stop `radar serve` before running CLI commands.

## Project layout

| File | Responsibility |
|---|---|
| `radar/arxiv_source.py` | arXiv API client (polite paging, PDF cache) |
| `radar/parsing.py` | PDF → sections |
| `radar/chunking.py` | Structure-aware chunking + contextual headers |
| `radar/store.py` | Qdrant (dense + sparse vectors) and SQLite metadata |
| `radar/ingest.py` | Idempotent, incremental ingestion pipeline |
| `radar/retrieval.py` | Dense, BM25, RRF fusion, cross-encoder rerank |
| `radar/agent.py` | Agentic loop: plan → route → retrieve ⇄ grade → answer → verify |
| `radar/trends.py` | Statistical trend detection |
| `radar/evaluation.py` | Synthetic test set, retrieval ablation, answer faithfulness |
| `radar/tracing.py` | Per-request step traces |
| `radar/api.py`, `radar/static/` | FastAPI + SSE streaming web UI |
| `radar/mcp_server.py` | Model Context Protocol server |

## Learn how it works

- [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md): every concept explained from scratch, mapped to the code
- [docs/HANDS_ON_TESTING.md](docs/HANDS_ON_TESTING.md): a testing lab, from unit tests to experiments that isolate each component; start with `uv run radar walkthrough "<question>"`
- [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md): pitch, design decisions, likely questions
- [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md): measured results
