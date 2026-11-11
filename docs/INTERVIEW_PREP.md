# Interview Prep — Research Radar

## Resume entry

**Research Radar — Agentic RAG for tracking AI research** · Python, Qdrant, Ollama (Qwen3-8B), FastAPI, MCP
*github.com/mdahmedanjarwala105/research-radar*

- Built an **agentic RAG system** over a continuously refreshed corpus of arXiv AI papers (400 papers / 2.6K chunks), with an LLM planner that routes queries, rewrites them into multiple searches, grades evidence and re-searches when it is insufficient (Corrective RAG). It abstains instead of hallucinating when evidence is missing.
- Implemented **hybrid retrieval** (dense `bge-small` + BM25 sparse vectors with Reciprocal Rank Fusion) and **cross-encoder reranking**. An ablation on 80 synthetic queries showed it is the only configuration robust to both query styles: **MRR 0.82 vs 0.62** for dense-only on name-specific queries, and **hit@8 0.53 vs 0.28** for BM25-only on plain-language queries.
- Enforced **grounded generation**: every claim carries an inline citation, and a verifier pass plus programmatic checks flag unsupported claims; **88% of end-to-end answers were fully supported** by their sources, and 100% cited the correct source paper.
- Designed a **statistical trend radar** (phrase-lift over time windows) so the LLM narrates measured trends instead of guessing them; surfaced rising topics such as *evidence selection* and *structured knowledge retrieval*.
- Shipped as a CLI, a streaming web UI (SSE), a REST API and an **MCP server**, so Claude, Cursor and other AI assistants can use it as a tool. Runs **fully locally** (no API keys, private by design), with idempotent, incremental ingestion for scheduled refresh and per-request tracing.

*Shorter, 2-bullet version:*
- Built a fully local **agentic RAG** system over live arXiv research (hybrid dense+BM25 retrieval, RRF, cross-encoder reranking, corrective re-retrieval, cited and verified answers), exposed via streaming UI, REST and an **MCP server**.
- Built an **evaluation harness** (two synthetic test sets, retrieval ablations, faithfulness checks): hybrid+rerank reached **0.82 MRR vs 0.62** for dense-only, nearly **2× BM25's recall** on plain-language queries, and **88%** of answers were fully grounded.

---

## 30-second pitch

> "LLMs are frozen at their training date, and AI research moves every week. So I built Research
> Radar: it ingests new arXiv papers on a schedule and answers questions about them with cited,
> verified answers, and it can tell you which research directions are rising or fading. It goes
> beyond basic RAG: hybrid search with reranking, an agent loop that grades its own evidence and
> re-searches when it's not enough, and an eval harness that measures every component. It runs
> fully locally on an 8B model, and it's exposed as an MCP server, so any AI assistant can use it."

## 2-minute technical walkthrough

1. **Ingestion.** arXiv API → PDF → section-aware chunks with a contextual header (paper + section).
   Each chunk is stored with a dense and a sparse vector in one Qdrant collection. Ids are
   deterministic, so ingestion is idempotent; `refresh` is incremental.
2. **Retrieval.** Dense finds paraphrases, BM25 finds exact terms like model names. I fuse them
   with RRF, because their scores aren't comparable, then a cross-encoder reranks the top 30.
3. **Agent.** The planner (constrained to a JSON schema) routes to search, trends or direct. For
   search it creates query variants, retrieves, then grades the passages. If they're insufficient,
   it writes a follow-up query, bounded to 2 rounds. Then it generates a cited answer and verifies it.
4. **Trends.** Computed statistically: share of papers mentioning a phrase, early vs recent half
   of the window. The LLM only narrates.
5. **Evaluation.** Synthetic questions with known gold chunks → hit@k and MRR across 4 retrieval
   modes, plus end-to-end faithfulness.
6. **Serving.** SSE streaming of agent steps and tokens, REST, and an MCP server.

---

## Likely questions and strong answers

**Q: Why not just use a bigger context window instead of RAG?**
Even with long contexts: (1) cost and latency grow with every token, on every query; (2) models
attend worse to the middle of very long contexts ("lost in the middle"); (3) my corpus grows daily
and won't fit; (4) retrieval gives citations, so answers can be checked. RAG and long context are
complementary: retrieve the best 8 chunks, then use the context window for reasoning.

**Q: Why hybrid search?**
Dense embeddings capture meaning but blur rare exact tokens: a model name like "Qwen3", an acronym
like "CRAG", a dataset name. BM25 nails those but misses paraphrases. I measured it with two test
sets. On questions that name the method, BM25 alone had the best MRR (0.86), even beating my
full pipeline (0.82). On plain-language questions BM25 collapsed to 0.28 hit@8, while
hybrid+rerank got 0.53. Users ask both kinds of question, so hybrid is the only robust choice.
The first result surprised me, and it taught me to test with more than one style of query.

**Q: Why RRF instead of a weighted sum of scores?**
Cosine similarity and BM25 live on different, query-dependent scales, so a weighted sum needs
normalisation and tuning per corpus. RRF only uses ranks, needs no tuning, and is robust. k=60
comes from the original paper and damps the gap between rank 1 and rank 2.

**Q: What does the reranker add, and what does it cost?**
A cross-encoder reads the query and passage together, so it captures interactions a bi-encoder
can't, which improves precision at the top. It costs one forward pass per candidate, so I only
rerank 30 candidates. That's the classic retrieve-then-rerank trade-off: cheap recall, then
expensive precision.

**Q: How do you prevent hallucinations?**
Layers: (1) answer only from passages, with inline citations; (2) evidence grading drops
irrelevant passages; (3) abstain when nothing relevant is found; (4) a verifier pass checks claims
against passages; (5) programmatic checks for invalid citation numbers and citation coverage;
(6) I don't trust the verifier's boolean alone: the code marks an answer unsupported if any claim
is flagged; (7) the planner can't answer from memory if the papers cover the question: code runs a
quick search and overrides the "direct" route when the reranker finds a real match. And for
trends, numbers come from statistics, not the LLM.

**Q: Why build the agent loop yourself instead of using LangChain or LangGraph?**
The loop is about 200 lines and fully transparent. I can trace, test and explain every step.
Frameworks are great for speed, but I wanted to understand what they do under the hood. The design
maps directly onto a LangGraph state graph if a team standardises on it.

**Q: How do you make an 8B model reliable enough for agent decisions?**
Structured outputs: every decision is a Pydantic schema, and Ollama enforces it with constrained
decoding, so the output is always valid JSON. Plus small, single-purpose prompts, temperature 0
for decisions, and deterministic guardrails around the model. Example: the planner sometimes
invented a 30-day date filter for the word "recent", so the code only honours a time window if
the question mentions a time unit.

**Q: How did you evaluate it? Isn't a synthetic test set biased?**
I sample chunks, have the LLM write a question each one answers, and measure whether retrieval
returns that gold chunk (hit@k, MRR) across four modes. Two honest caveats: synthetic questions
reuse source wording, which favours BM25, so I prompt for paraphrasing; and the same model
generates and judges. In production I'd add human-labelled queries from real usage and a stronger
judge model.

**Q: What was the hardest bug?**
When the evidence was graded "insufficient", the 8B model ignored the system prompt and wrote a
6,800-character uncited essay in 90 seconds. Fixes: repeat the rules at the *end* of the prompt
(small models weight recent instructions most), tell it explicitly what's missing so it can say
so, and cap the output length. Answers then dropped to ~150 words with 100% citation coverage.
The fix came from reading the traces, which is why I log every step.

**Q: Tell me about a time your system hallucinated.**
I asked it "EAEV", an acronym from one paper in the index. BM25 found the only chunk containing
it and the reranker put that paper first with +3.13, while everything else scored below −8.6. But
the planner had routed the question to "answer from memory", and the model invented "Empirical
Analysis of Economic Value". The retrieval was right; the router skipped it. I fixed it in two
layers: a stricter route definition, and a code guardrail. Before answering from memory, run one
sub-second search; if the top rerank score is above 0, switch to search. Small talk scores around
−7 to −10 and real topics +3 or more, so the threshold sits in a wide gap. I added unit tests with
a fake LLM that forces the bad route, so the guardrail is tested without depending on the model.

**Q: How would you scale this to millions of documents and many users?**
Qdrant server or cluster instead of embedded mode (one config change); ingestion as a queue-based
worker; GPU or batched embedding; a semantic cache for repeated questions; serve the LLM with vLLM,
or swap the `LLM` class to an API model; HNSW tuning and payload indexes for filters (already
indexed by date, kind and paper id).

**Q: How is this useful in the future, not just now?**
The corpus never goes stale: incremental refresh runs on a schedule. The trend radar gets *more*
valuable as history accumulates. And through MCP it's a building block for AI agents: a coding
or research agent can call `search_papers` to ground itself in current research.

**Q: What would you improve next?**
Multimodal parsing (tables and figures), GraphRAG over methods/datasets/citations,
embedding-based topic clustering for trends, a human-labelled eval set, and a user feedback loop
that feeds new eval cases.

---

## Numbers to remember

| Metric | Value |
|---|---|
| Corpus | 400 papers (40 full-text), 2,584 chunks, Aug–Oct 2026 |
| Chunk size | ~180 words, 2-sentence overlap, contextual header |
| Candidates | 30 per retriever per query → RRF → rerank → top 8 (≤3 per paper) |
| Specific questions (n=40): hybrid+rerank / dense / BM25 | MRR 0.82 / 0.62 / 0.86 · hit@3 0.98 / 0.68 / 0.93 |
| Conceptual questions (n=40): hybrid+rerank / dense / BM25 | hit@8 0.53 / 0.40 / 0.28 · MRR 0.25 / 0.20 / 0.14 |
| Right paper in top 8 (conceptual) | 80% (exact chunk is harder: many chunks answer a vague question) |
| Answer faithfulness | 88% (7/8; 8/8 on an earlier run, so small sample) · gold paper cited 100% |
| Latency | median ~20 s end to end on a laptop (mostly LLM generation, which is why it streams); retrieval ~0.7 s incl. rerank |
| Models | Qwen3-8B (Ollama), bge-small-en-v1.5, BM25, ms-marco-MiniLM-L-6 cross-encoder |
