# Research Radar — Learning Guide

This guide teaches you everything inside this project, from zero. Read it in
order once; afterwards use it as a reference. Every concept links to the file
that implements it, so you can read the real code right after the idea.

---

## Part 1 — Why RAG exists

A Large Language Model (LLM) like Qwen, Llama, GPT or Claude is a giant
next-word predictor trained on a snapshot of the internet. That causes three
problems:

1. **Frozen knowledge.** It knows nothing published after its training cutoff.
   AI research moves weekly, so a model trained last year has never heard of
   this month's papers.
2. **Hallucination.** When it doesn't know, it often *guesses fluently*:
   inventing paper titles, numbers, authors.
3. **No private data.** It has never seen your company's documents.

**Retrieval-Augmented Generation (RAG)** fixes this with an open-book exam:
before answering, *search* a document collection for relevant passages and
paste them into the prompt with the instruction "answer only from these".

```
question ──► search your documents ──► top passages ──► LLM writes answer citing them
```

The model no longer needs to *know* the answer; it needs to *read* well,
which LLMs are very good at.

---

## Part 2 — Core concepts from scratch

### 2.1 Tokens
LLMs read text as **tokens**, roughly ¾ of a word each. "Retrieval" might
be one token, "hallucination" two. Model speed is measured in tokens per
second (our local qwen3:8b does ~23 tok/s on your Mac), and the prompt
length limit (the "context window") is counted in tokens.

### 2.2 Embeddings: turning meaning into numbers
An **embedding model** converts text into a list of numbers (a *vector*).
Ours, `bge-small-en-v1.5`, outputs 384 numbers. It is trained so that texts
with similar *meaning* get similar vectors:

```
"the model makes things up"        → [0.12, -0.40, 0.88, …]
"LLM hallucination"                → [0.10, -0.37, 0.91, …]   ← close!
"recipe for banana bread"          → [-0.71, 0.22, -0.05, …]  ← far away
```

Notice the first two share **no words** but end up close. That is the magic
of *semantic* search.

### 2.3 Cosine similarity
How "close" are two vectors? Picture each one as an arrow from the origin.
**Cosine similarity** measures the angle between the arrows: 1.0 means the
same direction (same meaning), 0 means unrelated. Vector search = "find the
stored vectors with the highest cosine similarity to the query vector".

### 2.4 Vector database
Comparing a query against millions of vectors one by one is slow. A **vector
database** (we use **Qdrant**) stores vectors with an index (HNSW, a
navigable graph) that finds the nearest neighbours in milliseconds. It also
stores a **payload** (the chunk text, title, date) and can **filter** on it
("only papers from the last 30 days").
→ [`radar/store.py`](../radar/store.py)

### 2.5 BM25: classic keyword search
Before embeddings, search engines used **BM25** (and still do). It scores a
document higher when:
- it contains the query words **often** (term frequency, with diminishing returns),
- those words are **rare** across the collection (inverse document frequency, IDF):
  matching "ColBERT" is far more informative than matching "the",
- the document isn't just long (length normalisation).

BM25 is excellent at exact names, acronyms and numbers, which embeddings
blur. We store BM25 as a **sparse vector** (mostly zeros: one slot per
vocabulary word) right next to the dense embedding.

### 2.6 Chunking
We don't embed whole papers: one vector for 20 pages averages away every
specific idea. We split papers into **chunks** of ~180 words, so search can
pinpoint the exact paragraph.
→ [`radar/chunking.py`](../radar/chunking.py)

---

## Part 3 — The architecture

```
                         ┌──────────────── INGESTION (offline, scheduled) ────────────────┐
 arXiv API ──► papers ──►│ PDF ─► sections ─► chunks + contextual header ─► dense + BM25   │──► Qdrant
                         │                                                    vectors      │    SQLite
                         └─────────────────────────────────────────────────────────────────┘

                         ┌──────────────────── QUERY (online, per question) ───────────────┐
 question ──► PLAN ──────┤ route: search │ trends │ direct                                 │
                         │                                                                 │
   search ─► multi-query ─► dense + BM25 ─► RRF fusion ─► cross-encoder rerank ─► GRADE ───┤
                ▲                                                                   │      │
                └──────────── rewrite query if evidence insufficient ◄──────────────┘      │
                         │                                                                 │
   trends ─► statistics over paper counts per week ─► LLM narrates the numbers             │
                         │                                                                 │
                         └──► ANSWER with [n] citations (streamed) ──► VERIFY claims ──────┘
                                                                         │
                       CLI  ·  Web UI (SSE streaming)  ·  REST API  ·  MCP server
```

### Follow one question through the code

Question: *"How are recent papers reducing hallucination in RAG?"*

| Step | What happens | File |
|---|---|---|
| 1 | The planner LLM returns JSON: `route="search"`, queries like `["hallucination mitigation retrieval-augmented generation", "faithfulness grounding RAG"]` | [`agent.py`](../radar/agent.py) `Plan` |
| 2 | Each query runs through dense + BM25 search (30 candidates each) | [`retrieval.py`](../radar/retrieval.py) |
| 3 | All candidate lists are merged with Reciprocal Rank Fusion | `rrf()` |
| 4 | A cross-encoder re-scores the top 30 and we keep 8, at most 3 per paper | `rerank()`, `limit_per_paper()` |
| 5 | The LLM grades the 8 passages: which are relevant? Is it enough? | `Grade` |
| 6 | If not enough, it writes a follow-up query and we search again (max 2 rounds) | corrective loop |
| 7 | The LLM writes the answer using only the kept passages, citing `[n]` | `ANSWER_SYS` |
| 8 | A verifier LLM call checks every claim against the passages | `Verification` |
| 9 | Timings for every step are written to `data/traces/*.jsonl` | [`tracing.py`](../radar/tracing.py) |

---

## Part 4 — Each component, in depth

### 4.1 Ingestion — [`ingest.py`](../radar/ingest.py), [`arxiv_source.py`](../radar/arxiv_source.py)
- Uses the free **arXiv API**, waiting 3 s between calls as arXiv asks.
- Every paper gets its **abstract** indexed. The newest N also get **full
  text** (download PDF, parse). Two tiers keep it fast: broad coverage from
  abstracts, depth on recent papers.
- **Idempotent**: chunk ids are UUIDs derived from `paper_id + position`, so
  re-ingesting overwrites rather than duplicates.
- **Incremental**: `radar refresh` only fetches papers newer than the newest
  stored one. Put it in a nightly cron job and the radar stays current forever.

> *Interview line:* "Ingestion is idempotent and incremental, so it can run
> on a schedule without duplicating data or re-processing old papers."

### 4.2 PDF parsing — [`parsing.py`](../radar/parsing.py)
PDFs store positioned characters, not paragraphs. We extract text with
PyMuPDF, re-join hyphenated words, detect headings with a regex
(`3.2 Method`, `Conclusion`) and stop at `References`. Simple, fast and
explainable. Production systems sometimes use layout models (e.g. Docling,
Nougat) to also extract tables and equations; that's a listed upgrade.

### 4.3 Chunking + contextual headers — [`chunking.py`](../radar/chunking.py)
Rules: never cross a section boundary, split on sentences, ~180 words, and
**overlap** the last 2 sentences into the next chunk so ideas on the
boundary aren't cut in half.

Then the important trick: a chunk like *"Our method improves recall by 12%"*
is meaningless alone. Whose method? So before embedding we prepend:

```
Paper: TriageRAG: Failure-Aware Self-Training for ...
Section: 4 Experiments

Our method improves recall by 12% ...
```

This is a lightweight version of **Contextual Retrieval** (Anthropic, 2024),
which reported large drops in retrieval failures from adding context to chunks.
Anthropic's full version asks an LLM to write the context for every chunk;
on a laptop with a local model that would take hours for thousands of
chunks, so we use the free, deterministic header.

> *Trade-off to mention:* chunk size. Small chunks give precise retrieval
> but lose context; large chunks keep context but dilute the embedding and
> waste prompt space. 150–300 words is a common sweet spot, and the eval
> harness lets you measure it (see exercises).

### 4.4 Hybrid search + RRF — [`retrieval.py`](../radar/retrieval.py)
Dense search finds paraphrases; BM25 finds exact terms. **Hybrid search**
runs both. But their scores can't be added: cosine ≈ 0.3–0.9, BM25 ≈ 0–40.

**Reciprocal Rank Fusion** ignores scores and uses only *ranks*:

```
RRF(doc) = Σ over each result list  1 / (k + rank)          k = 60
```

Worked example:

| chunk | dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | — | 1/61 = 0.0164 |
| B | 2 | 1 | 1/62 + 1/61 = **0.0325** ← wins |
| C | 3 | — | 1/63 = 0.0159 |

B wins because *both* retrievers agree it's good. The same function also
merges results from multiple query rewrites ("multi-query", or RAG-Fusion).
Our unit test `test_rrf_rewards_agreement_between_retrievers` checks exactly
this example.

### 4.5 Reranking — `Retriever.rerank()`
There are two kinds of models that score relevance:

| | **Bi-encoder** (embedding model) | **Cross-encoder** (reranker) |
|---|---|---|
| How | encodes query and chunk *separately*, compares vectors | reads query + chunk *together* in one pass |
| Speed | chunk vectors precomputed, so search is instant | must run once per (query, chunk) pair |
| Accuracy | good | better: sees word-level interactions |
| Used for | stage 1: recall over everything | stage 2: precision on ~30 candidates |

This **retrieve-then-rerank** two-stage design is standard in search
engines. Our reranker `ms-marco-MiniLM-L-6-v2` is small enough for a CPU.

### 4.6 The agent — [`agent.py`](../radar/agent.py)
"Agentic RAG" means the LLM **makes decisions** in the loop instead of
running a fixed pipeline once. Ours is an explicit state machine, where
the LLM makes small decisions between deterministic steps:

1. **Plan and route** (Adaptive-RAG idea): is this a paper-content question
   (`search`), a "what's trending" question (`trends`), or a general one
   (`direct`)? It also rewrites the question into better search queries and
   detects time limits ("this month" → 30 days → a date filter in Qdrant).
2. **Grade evidence** (Corrective RAG / CRAG idea): the LLM marks which
   passages are actually relevant and whether they're *sufficient*.
3. **Corrective retry**: if insufficient, the LLM writes a follow-up query
   for the missing piece, and we search again (bounded to 2 rounds so it
   can't loop forever).
4. **Abstain**: if nothing relevant is found, it says so instead of guessing.
   Knowing when *not* to answer is a key production feature.
5. **Cited answer**: every claim carries `[n]` pointing to a passage.
6. **Verify** (Self-RAG-style self-critique): a second LLM call checks each
   claim against the passages, and we programmatically check that every `[n]`
   refers to a real source.

**Why structured outputs matter.** Small 8B models are unreliable at
free-form tool calling. Every decision here uses a Pydantic schema, which
Ollama enforces with **constrained decoding**: at each step the model may
only produce tokens that keep the JSON valid. Unparseable decisions become
impossible, which is what makes an 8B model usable as an agent.

**Why not LangChain/LangGraph?** They're fine, but writing the loop by hand
(about 200 lines) means you can explain every line in an interview. Saying
"I understand what the frameworks do under the hood" is a strength.

### 4.7 Trends — [`trends.py`](../radar/trends.py)
Ask an LLM "what's trending?" and it will confidently make something up.
So trends are computed with **statistics**, and the LLM only narrates the
numbers:
- extract 2–3 word phrases from titles and abstracts (dropping stop-words),
- split the time window into early and recent halves,
- **lift** = (share of recent papers mentioning the phrase) − (share of early papers),
- require minimum support (≥3 papers) so one paper can't create a "trend".

The design principle, *LLMs for language, code for facts*, is worth
saying out loud in interviews.

### 4.8 Evaluation — [`evaluation.py`](../radar/evaluation.py)
You can't improve what you don't measure.

1. **Synthetic test set**: sample chunks, have the LLM write a question each
   chunk answers. Now each question has a known "gold" chunk, and nobody had
   to label anything by hand.
2. **Retrieval metrics**:
   - **hit@k**: did the gold chunk appear in the top k? (k = 8)
   - **MRR** (Mean Reciprocal Rank): average of 1/rank. Rank 1 gives 1.0,
     rank 2 gives 0.5, rank 4 gives 0.25, not found gives 0. Rewards ranking it *high*.
   - Example: ranks for 4 questions = 1, 3, not found, 2 → MRR = (1 + 0.33 + 0 + 0.5) / 4 = 0.46
3. **Ablation**: run the same questions through `dense`, `sparse`, `hybrid`,
   `hybrid_rerank` and compare. This *proves* each stage adds value.
   Results: [`EVAL_RESULTS.md`](EVAL_RESULTS.md).
4. **Answer quality**: faithfulness (verifier: are all claims supported?),
   and whether the gold paper got cited.

**What our numbers actually showed** (this is the most valuable thing in the project):

| mode | *specific* questions: hit@8 / MRR | *conceptual* questions: hit@8 / MRR |
|---|---|---|
| dense only | 0.88 / 0.62 | 0.40 / 0.20 |
| BM25 only | 0.98 / **0.86** | 0.28 / 0.14 |
| hybrid (RRF) | 0.95 / 0.79 | 0.53 / 0.22 |
| hybrid + rerank | **0.98** / 0.82 | **0.53 / 0.25** |

- The first test set had questions that named the method ("How does TriageRAG…").
  **BM25 won**, beating the full pipeline on MRR. A naive conclusion: "hybrid
  search is pointless".
- So we built a second set of *conceptual* questions in plain words, with no names
  ("How can a system ensure it doesn't make up details?"). **BM25 collapsed**
  (0.98 → 0.28 hit@8) because it had no exact words to match. Hybrid + rerank was best.
- Hybrid + rerank is the only configuration that's good on *both*. Real users ask
  both kinds of question, so that's why it's the production default.
- Conceptual scores are lower overall because a vague question has many valid
  answers but only one "gold" chunk. Paper-level hit@8 (80%) is the fairer view there.
- Reranking costs ~0.7 s per query, which is small next to ~20 s of LLM generation.

The lesson: **one benchmark can mislead you; test with the query styles your users
really use.**

Honest caveats (mentioning them shows maturity): synthetic questions often
reuse the chunk's wording, which flatters BM25; and the same 8B model
generates and judges. Production teams add human-labelled questions and a
stronger judge model.

### 4.9 Observability — [`tracing.py`](../radar/tracing.py)
Every request writes a **trace**: each step's name, inputs and duration in
milliseconds. When an answer is bad you can see *why*: was it retrieval
(wrong chunks) or generation (right chunks, bad answer)? Tools like Langfuse
and LangSmith do this at scale; ours is the same idea in 40 lines.

### 4.10 Serving: SSE and MCP — [`api.py`](../radar/api.py), [`mcp_server.py`](../radar/mcp_server.py)
- **Server-Sent Events (SSE)** stream each agent step and each answer token to
  the browser as they're produced. A 30-second answer feels interactive
  because the user watches the agent think.
- **MCP (Model Context Protocol)** is the open standard for connecting tools to
  AI assistants (Claude, ChatGPT, Cursor, VS Code…). Exposing Research Radar
  as an MCP server means *any* AI assistant can search your paper index.
  This is the "agents using tools" direction the industry is moving in.

---

## Part 5 — Naive vs advanced vs agentic RAG

| | Naive RAG (2023) | Advanced RAG | **Agentic RAG (this project)** |
|---|---|---|---|
| Search | dense only | hybrid + rerank | hybrid + rerank + multi-query |
| Chunks | fixed-size | structure-aware | structure-aware + contextual headers |
| Flow | retrieve once → answer | retrieve once → answer | plan → route → retrieve ⇄ grade/retry → answer → verify |
| When nothing is found | hallucinates | hallucinates | abstains |
| Citations | none | sometimes | every claim, verified |
| Data freshness | static upload | static upload | scheduled incremental ingestion |
| Quality | "looks good" | maybe some metrics | ablation eval harness + traces |
| Integration | a chat UI | an API | CLI + streaming UI + REST + **MCP** |

---

## Part 6 — Limitations and what you'd do next

Be ready to say these yourself. Interviewers like candidates who know where
their system breaks.

- **Local 8B model** → answers take ~20–60 s, and reasoning is weaker than
  frontier models. *Fix:* the `LLM` class is the only place that talks to the
  model; swap in a bigger local model or an API model.
- **Embedded Qdrant** → single process (the CLI and server can't open it at
  the same time). *Fix:* run the Qdrant server in Docker; it's a one-line change.
- **Figures and tables are ignored.** *Fix:* multimodal parsing (vision-language
  models, or a layout parser like Docling).
- **Trends use n-gram counts.** *Fix:* cluster embeddings per week (topic
  modelling, e.g. BERTopic) to catch trends phrased in different words.
- **Next-level retrieval ideas:** GraphRAG (build a graph of methods, datasets
  and citations), ColBERT late-interaction retrieval, semantic answer cache,
  user feedback (👍/👎) collected as new evaluation data.

---

## Part 6½ — Real bugs found while building this (and what they teach)

These happened during development. Each one is a ready-made interview story.

| Symptom (seen in the trace) | Root cause | Fix | Lesson |
|---|---|---|---|
| Planner wrote "2024" into search queries | The model doesn't know today's date | Put today's date in the planner prompt | LLMs have no clock; give them context explicitly |
| "Recent papers…" silently filtered results to the last 30 days | The 8B planner over-interprets "recent" | Code only accepts a time window if the question contains a time word | Wrap LLM decisions in deterministic guardrails |
| "What's trending?" went to plain search | Router prompt was vague | Explicit route definitions + examples in the prompt | Few-shot examples steer small models strongly |
| Insufficient evidence → 6,800-char uncited essay, 90 s | Small models drift on long prompts and forget system rules | Repeat rules at the *end* of the prompt, state what's missing, cap tokens | Instruction placement matters; always cap output |
| Verifier said `supported: true` *and* listed an unsupported claim | LLM self-contradiction | Code computes "supported" from the claim list | Never trust a model's summary boolean; derive it |
| Trend stats used only 48 of 320 papers | Topic filter reused search, capped at ~60 results | Rank all abstracts, keep those within 0.10 of the best match | Cosine scores are compressed and topic-dependent; use relative cutoffs |
| Trends full of "github com", "consistently outperforms" | Academic boilerplate | Domain-specific stop-words | Statistics are only as good as your features |

The general lesson: **log every step (traces), test with real questions, and fix
causes in code rather than hoping a prompt tweak holds.**

---

## Part 7 — Hands-on exercises (this is how you *really* learn it)

Run `uv run radar eval run --answers 0` after each change and compare.

1. **Chunk size:** set `RADAR_CHUNK_WORDS=100`, then 300. Re-ingest into a fresh
   data dir (`RADAR_DATA_DIR=data_small`), rebuild the test set, and compare hit@8.
2. **RRF constant:** try `RADAR_RRF_K=10` vs `60`. What happens to MRR?
3. **See BM25 vs dense disagree:** `uv run radar search "CRAG" --mode dense` vs
   `--mode sparse`. Then try a paraphrase like `"agent that double-checks its sources"`.
4. **Break the agent:** ask about something not in the index ("What did the
   2019 BERT paper do?"). Does it abstain or hallucinate? Read the trace.
5. **Swap the reranker:** `RADAR_RERANK_MODEL=BAAI/bge-reranker-base` (bigger).
   Is it worth the latency?
6. **Read a trace:** `tail -1 data/traces/*.jsonl | python -m json.tool`. Which
   step is slowest? (Spoiler: generation, which is why streaming matters.)

---

## Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model: predicts the next token; can follow instructions |
| **Token** | ~¾ of a word; the unit LLMs read and write |
| **Context window** | Max tokens an LLM can see at once (prompt + answer) |
| **Embedding** | A vector of numbers representing the meaning of a text |
| **Dense vector** | An embedding: every slot holds a value (384 here) |
| **Sparse vector** | Mostly zeros; one slot per vocabulary word (BM25 here) |
| **Cosine similarity** | Angle-based closeness of two vectors (1 = same direction) |
| **Vector DB** | Database for fast nearest-neighbour search over vectors (Qdrant) |
| **HNSW** | The graph index vector DBs use for fast approximate search |
| **BM25** | Classic keyword relevance formula (term frequency × rarity) |
| **Hybrid search** | Dense + keyword search combined |
| **RRF** | Reciprocal Rank Fusion: merge ranked lists using ranks only |
| **Bi-encoder** | Encodes query and doc separately (fast, used for search) |
| **Cross-encoder / reranker** | Reads query and doc together (accurate, slow) |
| **Chunk** | A ~paragraph-sized piece of a document |
| **Contextual retrieval** | Adding document-level context to each chunk before embedding |
| **Agentic RAG** | The LLM decides steps (route, retry, verify) instead of a fixed pipeline |
| **CRAG** | Corrective RAG: grade retrieved docs and re-search when they're poor |
| **Structured output / constrained decoding** | Forcing the LLM to emit JSON matching a schema |
| **Hallucination** | Fluent but false model output |
| **Faithfulness / groundedness** | Is every claim in the answer supported by the sources? |
| **hit@k / MRR** | Retrieval metrics: found in top k / how high it ranked |
| **Ablation** | Remove one component at a time to measure its contribution |
| **Trace** | Per-request log of every step and its timing |
| **SSE** | Server-Sent Events: server pushes a stream of messages to the browser |
| **MCP** | Model Context Protocol: open standard for plugging tools into AI assistants |
| **Ollama** | Runs open-weight LLMs locally behind a simple API |
| **Quantization** | Storing model weights in fewer bits (4-bit) so big models fit in laptop RAM |
