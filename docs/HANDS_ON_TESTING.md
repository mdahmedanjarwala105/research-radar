# Hands-On Testing Lab

How to test Research Radar so that you **understand** it, not just confirm it runs.
Each level answers a different question, and each experiment isolates one part of
the RAG so you can see what that part (and its code) does.

| Level | Question it answers | Time | Needs Ollama? |
|---|---|---|---|
| 1. Unit tests | Does each small piece of logic do what I think? | seconds | no |
| 2. Walkthrough | What happens to *one* question at every stage? | ~1 min | yes |
| 3. Experiments | What does each component contribute, and when does it fail? | 5–15 min | some |
| 4. Measurement | Is it good *on average*, and did my change help? | 2–10 min | optional |
| 5. Traces & UI | Where did the time go, and why did this answer happen? | minutes | yes |
| 6. Self-quiz | Can I explain it to an interviewer? | 10 min | no |

**Before you start** (in a terminal, from the project folder):

```bash
~/.local/ollama/ollama serve
```
```bash
uv run radar stats
```

`stats` should show ~400 papers and ~2,584 chunks. Stop `radar serve` if it's running:
only one program can open the index at a time.

---

## Level 1 — Unit tests: one function at a time

```bash
uv run pytest -v
```

| Test | RAG part | What it proves | Code |
|---|---|---|---|
| `test_rrf_rewards_agreement_between_retrievers` | Fusion | A chunk found by *both* retrievers beats one found by only one | `retrieval.py → rrf()` |
| `test_rrf_score_formula` | Fusion | The score really is `1/(k + rank)` | `rrf()` |
| `test_limit_per_paper` | Diversity | No single paper can hog all 8 slots | `limit_per_paper()` |
| `test_sentence_split_and_pack_with_overlap` | Chunking | Chunks break at sentences and overlap | `chunking.py → pack_sentences()` |
| `test_chunk_paper_adds_contextual_header_and_stable_ids` | Chunking | Header is added; ids are deterministic (so re-ingesting never duplicates) | `chunk_paper()`, `chunk_id()` |
| `test_heading_detection` | PDF parsing | "3.2 Method" is a heading; a normal sentence isn't | `parsing.py → _heading()` |
| `test_cited_numbers` | Citations | `[2, 3]` and `[3][4]` are both read correctly | `agent.py → cited_numbers()` |
| `test_phrases_drop_stopwords` | Trends | Phrases ignore filler words | `trends.py → phrases()` |

These need no AI models, so they run in about a second. That's the point: test logic
without slow, random LLMs.

### 🧪 Exercise 1.1 — Break the code on purpose ("mutation testing")
1. In [radar/retrieval.py](../radar/retrieval.py), inside `rrf()`, change
   `h.score += 1.0 / (k + rank)` to `h.score += 1.0 / rank`.
2. Run `uv run pytest -v`. **Which tests fail, and why?**
3. Undo the change (`git checkout radar/retrieval.py`).

When we tried it, only `test_rrf_score_formula` failed. The agreement test still
passed, because with `1/rank` a chunk found by both lists *still* beats one found by
one list (1/2 + 1/1 > 1/1). So that test alone would have missed the bug. This is how
you judge whether tests are any good: if no test fails when you break a line, nothing
is really checking that line.

### 🧪 Exercise 1.2 — Write your own test
Add this to [tests/test_core.py](../tests/test_core.py) and make it pass:

```python
def test_chunks_never_cross_sections():
    sections = [("1 Intro", "Alpha one. " * 40), ("2 Method", "Beta two. " * 40)]
    chunks = chunk_paper("x", "T", "Abstract.", sections)
    for c in chunks[1:]:
        # a chunk is either all-Intro text or all-Method text, never a mix
        assert not ("Alpha" in c.text and "Beta" in c.text)
```

---

## Level 2 — The walkthrough: follow one question through every stage

```bash
uv run radar walkthrough "How do papers reduce hallucinations in retrieval-augmented generation?"
```

Press Enter to move between stages. Every stage prints **what's happening**, **the real
data**, **the code that did it**, and a **Try:** idea.

| Stage | You'll see | What to notice |
|---|---|---|
| 1. Index | One stored chunk with its `Paper:/Section:` header | That header was embedded too; it's why a chunk "knows" its paper |
| 2. Embedding | 384 numbers; cosine 0.80 to the top chunk vs 0.44 to a banana-bread recipe | Unrelated text still scores 0.44: the *ranking* matters, not the raw number |
| 3. BM25 | Each word → stem → how many chunks contain it → IDF | `paper` has IDF **0.00**: our own `Paper:` header put it in every chunk, so BM25 learned it means nothing. `hallucin` (in 204 chunks) is worth 6× more than `retriev` (in 1,739) |
| 4. Two retrievers | Dense top-5 and BM25 top-5 side by side | They agree on only ~8 of their top 30. Each finds things the other misses |
| 5. RRF | The actual `1/(60+1) + 1/(60+3)` math per chunk | The fused #1 wasn't #1 in BM25; it won because *both* ranked it high |
| 6. Rerank | Old rank → new rank, with arrows | In our run the reranker pushed a **privacy** paper (RAG-CT) from #9 to #1. Rerankers make mistakes too… |
| 7. Agent | Every prompt sent to the LLM (magenta) and the JSON it returned (green) | …and the **grade** step then rejected that privacy paper. Layers of defense: a later step catches an earlier step's mistake |

Run it again with **your own** questions. Good ones to compare:
- A question using a paper's exact method name
- The same idea in plain everyday words
- Something clearly *not* in the index

---

## Level 3 — Experiments: isolate one component at a time

### 3.1 Dense vs BM25: when does each one fail?

```bash
uv run radar search "EAEV" --mode dense
```
```bash
uv run radar search "EAEV" --mode sparse
```
```bash
uv run radar search "EAEV" --mode hybrid_rerank
```

What we observed:

| Query | dense #1 | BM25 #1 | hybrid+rerank #1 |
|---|---|---|---|
| `EAEV` (an acronym) | ❌ a paper about *tractor operator manuals* | ✅ Evidence-Aligned Entity Verification | ✅ same |
| plain-words description of TriageRAG* | ✅ rank 1 | ⚠️ rank 6 | ✅ rank 1 |

\* *"a method where a question-answering system learns from its own mistakes to adapt to a new specialist field"*

**Lesson:** dense search doesn't know a made-up acronym, and BM25 doesn't know synonyms.
Hybrid covers both. This is the whole argument for hybrid search, in one experiment.

### 3.2 The RRF constant k

```bash
RADAR_RRF_K=5 uv run radar walkthrough "How do papers reduce hallucinations in retrieval-augmented generation?" --no-pause
```

Compare Stage 5 with the default (k=60). With small k, `1/(5+1)` is far bigger than
`1/(5+10)`, so being #1 in *one* list dominates. With k=60 the gap is tiny and
*agreement* between lists matters more. (`RADAR_…` variables override
[radar/config.py](../radar/config.py) for one run.)

### 3.3 Is the reranker worth it?

```bash
uv run radar search "how to stop a model from inventing facts" --mode hybrid
```
```bash
uv run radar search "how to stop a model from inventing facts" --mode hybrid_rerank
```

Do the top results get more relevant? The `found by` column shows where each came from.
The `rerank` column shows the cross-encoder's score.

### 3.4 Routing: three kinds of question

```bash
uv run radar ask "What does RAG stand for?"
```
```bash
uv run radar ask "What's trending in RAG research over the last two months?"
```
```bash
uv run radar ask "How do recent papers evaluate whether RAG answers are faithful?"
```

Look at the first line (`▸ plan`): the routes should be `direct`, `trends` and `search`.
The `direct` answer is labelled *"General knowledge: no papers were consulted"*.
Trends shows phrase counts (`early → recent`) that came from statistics, not the LLM.

### 3.5 Does it admit what it doesn't know?

```bash
uv run radar ask "What learning rate schedule did the original 2017 Transformer paper use?"
```

Expected: `grade … "kept": 0, "of": 8` and *"I couldn't find passages…"*. It does **not**
answer from memory, even though the LLM probably knows. Is that the right behaviour for
a research tool? (Discuss it in interviews: trust vs helpfulness.)

### 3.6 The time-filter guardrail

```bash
uv run radar ask "What did papers this week say about evaluation?" --no-verify
```
```bash
uv run radar ask "What do recent papers say about evaluation?" --no-verify
```

Check `recent_days` in the plan line: it should be `7` for the first and `null` for the
second. The code in `agent.py` (look for `TIME_WORDS`) overrides the LLM if it invents
a time window. That's an LLM decision wrapped in a deterministic rule.

### 3.7 Watch the corrective loop
Ask a two-part question the index can only partly answer, e.g.
*"How do papers detect hallucinations, and how much do these methods cost to run?"*
If `grade` says `"sufficient": false`, you'll see a **round 2** `retrieve` with a new
query written by the LLM. That's Corrective RAG.

---

## Level 4 — Measure: is it good on average?

Single questions are anecdotes. The eval harness runs 80 questions with known answers.

```bash
uv run radar eval run --answers 0
```

(`--answers 0` skips the slow LLM part: retrieval only, ~1–2 minutes.) It prints two
tables, one for questions that name the method and one for plain-words questions:

- **hit@k**: did the right chunk appear in the top k?
- **MRR**: how high, on average? (#1 → 1.0, #2 → 0.5, #4 → 0.25, missing → 0)

### 🧪 Exercise 4.1 — Change one thing, measure, explain
Run each and compare with [EVAL_RESULTS.md](EVAL_RESULTS.md):

```bash
RADAR_RRF_K=5 uv run radar eval run --answers 0
```
```bash
RADAR_CANDIDATES_PER_RETRIEVER=10 uv run radar eval run --answers 0
```
```bash
RADAR_RERANK_MODEL=BAAI/bge-reranker-base uv run radar eval run --answers 0
```

For each: did hit@8 or MRR go up or down? On which question style? Was the slowdown
(`p50_ms`) worth it? **Being able to say "I tried X, it changed Y by Z, so I kept/dropped
it" is exactly what interviewers want to hear.**

> These runs overwrite `docs/EVAL_RESULTS.md`. Restore it with
> `git checkout docs/EVAL_RESULTS.md` (your resume numbers come from that file).

### Exercise 4.2 (bigger) — Chunk size
Chunk size is fixed at ingestion time, so test it in a separate data folder:

```bash
RADAR_DATA_DIR=data_small RADAR_CHUNK_WORDS=90 uv run radar ingest --topic "retrieval augmented generation" --max-papers 100 --fulltext 15
```

Then build and run the eval with the same `RADAR_DATA_DIR`/`RADAR_CHUNK_WORDS` prefix, and
again with `RADAR_CHUNK_WORDS=300` in a third folder. Smaller chunks: more precise, but
less context. Which wins here?

---

## Level 5 — Traces and the web UI: why did *this* answer happen?

Every question appends a trace (each step, its inputs and how long it took). To see the latest:

```bash
ls -t data/traces/*.jsonl | head -1 | xargs tail -1 | python3 -m json.tool
```

Typical result: retrieval takes **under 1 s**, while every LLM step (plan, grade,
generate, verify) takes **3–12 s**. That's why the UI streams the answer, and why "use a
bigger model" has a real latency cost.

**Debugging recipe** when an answer is bad: open its trace and ask:
1. Did `retrieve` return the right chunks? If not, it's a **retrieval** problem
   (try `radar search` with the same query).
2. Did `grade` keep the right ones? If not, it's a **grading** problem.
3. Were the right passages there, but the answer still wrong? That's a **generation** problem.

The web UI shows the same steps live, in the "Agent trace" panel:

```bash
uv run radar serve
```

Then open http://127.0.0.1:8000.

---

## Level 6 — Self-quiz (answer out loud, then check)

<details><summary>1. Why does the word "paper" have an IDF of 0 in our BM25 index?</summary>
Every chunk starts with the header "Paper: …", so the stem appears in all 2,584 chunks.
A word in every document can't distinguish any of them, so its rarity weight is zero.
That's a side effect of contextual headers on keyword search.
</details>

<details><summary>2. Dense search ranked a tractor-manual paper #1 for "EAEV". Why?</summary>
The embedding model has never seen the made-up acronym "EAEV", so its vector is
essentially noise, and the nearest chunk is random-ish. BM25 matches the exact token,
which appears in only one paper.
</details>

<details><summary>3. Why can't we just add the dense score and the BM25 score?</summary>
They're on different scales (cosine ~0.4–0.9 vs BM25 ~0–15, varying per query). RRF uses
ranks only, so no normalisation or tuning is needed.
</details>

<details><summary>4. The reranker promoted a wrong paper to #1. Why did the final answer still not cite it?</summary>
The grade step (an LLM judging relevance) rejected it. Each layer catches some of the
previous layer's mistakes.
</details>

<details><summary>5. Why use unit tests AND an eval harness?</summary>
Unit tests check deterministic logic (is RRF computed correctly?). The eval harness
checks quality (does the system find the right answers on average?). A system can pass
every unit test and still retrieve badly.
</details>

<details><summary>6. An answer is wrong. How do you find out which component failed?</summary>
Open its trace. Check retrieval (right chunks returned?), then grading (right ones
kept?), then generation (right passages, wrong answer?). Fix the first stage that failed.
</details>
