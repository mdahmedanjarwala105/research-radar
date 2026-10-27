"""Evaluation: prove each component earns its place, with numbers.

"It seems to work" is not engineering. Here we:

1. BUILD A TEST SET synthetically: sample chunks from the index and have the
   LLM write a question that chunk answers. We now know the "gold" chunk for
   each question, without hand-labelling anything.

2. RETRIEVAL EVAL (no LLM needed, fast): for each retrieval mode, check
   whether the gold chunk comes back.
     - hit@k : fraction of questions whose gold chunk is in the top k
     - MRR   : mean of 1/rank of the gold chunk (rewards ranking it high)
     - paper_hit@k : gold *paper* in top k (a looser, user-facing view)
   Comparing dense vs sparse vs hybrid vs hybrid+rerank is an ablation
   study: it shows what each stage contributes.

3. ANSWER EVAL (slow, runs the full agent): is the answer grounded in its
   sources (faithfulness, judged by the verifier), and does it cite the
   gold paper?

Known caveats (say these in interviews, it shows maturity):
- Synthetic questions tend to reuse the chunk's words, which flatters BM25.
  The generation prompt asks for paraphrasing to reduce this.
- The same LLM writes and judges answers; a stronger judge model would be
  more trustworthy. Real deployments add human-labelled examples.
"""

import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from radar.agent import ResearchAgent
from radar.config import settings
from radar.llm import LLM
from radar.retrieval import MODES, Retriever
from radar.store import Store


class SynthQuestion(BaseModel):
    question: str = Field(description="The question")


QGEN_PROMPTS = {
    # Mimics a reader who knows what they're looking for (names, methods, datasets).
    "specific": """You write evaluation questions for a search engine over AI research papers.
Given a passage, write ONE question that a researcher might ask and that this passage answers.
Rules:
- Paraphrase: do not copy distinctive phrases from the passage word-for-word.
- Be specific enough that this passage (not just any paper) answers it: mention the method,
  dataset or concept by name when the passage gives one.
- Never say "the passage", "this paper" or "the authors".
Output JSON only.""",
    # Mimics a user who knows the idea but not the paper's vocabulary: the hard,
    # realistic case where keyword search has nothing exact to match.
    "conceptual": """You write evaluation questions for a search engine over AI research papers.
Given a passage, write ONE question, asked by someone who has NOT read the paper, that this passage answers.
Rules:
- Describe the idea in plain everyday words.
- Do NOT mention any method name, model name, dataset, benchmark, acronym or number.
- Do NOT reuse distinctive technical phrases from the passage; use synonyms.
- Never say "the passage", "this paper" or "the authors".
Output JSON only.""",
}
STYLES = tuple(QGEN_PROMPTS)


def eval_dir() -> Path:
    d = settings.data_dir / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def testset_path(style: str) -> Path:
    return eval_dir() / f"testset_{style}.jsonl"


def build_testset(
    store: Store, llm: LLM, n: int = 40, style: str = "specific", seed: int = 7, log=print
) -> Path:
    rng = random.Random(seed)
    full = [p for p in store.scroll_chunks(kind="fulltext") if len(p.payload["text"].split()) >= 80]
    abstracts = list(store.scroll_chunks(kind="abstract"))
    n_full = min(len(full), round(n * 0.75))
    picked = rng.sample(full, n_full) + rng.sample(abstracts, min(len(abstracts), n - n_full))

    path = testset_path(style)
    with path.open("w") as f:
        for i, p in enumerate(picked, 1):
            pl = p.payload
            q = llm.structured(
                QGEN_PROMPTS[style],
                f"Paper: {pl['title']}\nSection: {pl['section']}\n\nPassage:\n{pl['text']}",
                SynthQuestion,
                temperature=0.3,
            ).question
            f.write(json.dumps({"question": q, "chunk_id": str(p.id), "paper_id": pl["paper_id"],
                                "kind": pl["kind"], "section": pl["section"], "style": style}) + "\n")
            log(f"[{i}/{len(picked)}] {q}")
    return path


def load_testset(style: str = "specific") -> list[dict]:
    path = testset_path(style)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def available_styles() -> list[str]:
    return [s for s in STYLES if testset_path(s).exists()]


def retrieval_eval(retriever: Retriever, testset: list[dict], modes=MODES, k: int = 8) -> dict:
    results = {}
    for mode in modes:
        ranks, paper_hits, latencies = [], [], []
        for row in testset:
            t0 = time.perf_counter()
            hits = retriever.search(row["question"], mode=mode, k=k, max_per_paper=k)
            latencies.append((time.perf_counter() - t0) * 1000)
            ids = [h.id for h in hits]
            ranks.append(ids.index(row["chunk_id"]) + 1 if row["chunk_id"] in ids else None)
            paper_hits.append(row["paper_id"] in {h.paper_id for h in hits})
        n = len(testset)
        results[mode] = {
            "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
            "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
            f"hit@{k}": round(sum(r is not None for r in ranks) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            f"paper_hit@{k}": round(sum(paper_hits) / n, 3),
            "p50_ms": round(statistics.median(latencies)),
        }
    return results


def answer_eval(agent: ResearchAgent, testset: list[dict], n: int = 8, log=print) -> dict:
    rows = []
    for i, row in enumerate(testset[:n], 1):
        out = agent.ask(row["question"], verify=True)
        v = out.get("verification") or {}
        cited_papers = {s["paper_id"] for s in out["sources"] if s.get("cited")}
        rows.append({
            "question": row["question"],
            "faithful": bool(v.get("supported")),
            "cites_gold_paper": row["paper_id"] in cited_papers,
            "abstained": not out["sources"],
            "seconds": round(out["total_ms"] / 1000, 1),
        })
        log(f"[{i}/{n}] faithful={rows[-1]['faithful']} gold_cited={rows[-1]['cites_gold_paper']} "
            f"{rows[-1]['seconds']}s  {row['question'][:70]}")
    m = len(rows)
    return {
        "n": m,
        "faithfulness": round(sum(r["faithful"] for r in rows) / m, 3),
        "gold_paper_cited": round(sum(r["cites_gold_paper"] for r in rows) / m, 3),
        "abstain_rate": round(sum(r["abstained"] for r in rows) / m, 3),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "rows": rows,
    }


STYLE_NOTES = {
    "specific": "Questions name the method/dataset (a reader who knows what they want). "
    "Exact names favour BM25 keyword search.",
    "conceptual": "Questions describe the idea in plain words with no names or acronyms "
    "(a user who doesn't know the paper's vocabulary). This tests semantic retrieval.",
}


def save_report(retrieval_by_style: dict[str, dict], answers: dict | None, sizes: dict[str, int],
                k: int, stats: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (eval_dir() / "results.json").write_text(json.dumps(
        {"time": stamp, "retrieval": retrieval_by_style, "answers": answers, "corpus": stats}, indent=2))

    lines = [
        "# Evaluation results",
        "",
        f"_Generated {stamp} on {stats['papers']} papers / {stats['chunks']} chunks, k={k}. "
        f"Models: `{settings.embed_model}`, `{settings.sparse_model}`, `{settings.rerank_model}`, "
        f"`{settings.llm_model}`._",
    ]
    for style, retrieval in retrieval_by_style.items():
        cols = list(next(iter(retrieval.values())).keys())
        lines += [
            "",
            f"## Retrieval ablation — {style} questions (n={sizes[style]})",
            "",
            f"_{STYLE_NOTES.get(style, '')}_",
            "",
            "| mode | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for mode, row in retrieval.items():
            lines.append(f"| {mode} | " + " | ".join(str(row[c]) for c in cols) + " |")
    if answers:
        lines += [
            "",
            "## End-to-end answers",
            "",
            f"- Questions: {answers['n']}",
            f"- Faithfulness (verifier says every claim is supported): {answers['faithfulness']:.0%}",
            f"- Cites the gold paper: {answers['gold_paper_cited']:.0%}",
            f"- Abstained (no evidence found): {answers['abstain_rate']:.0%}",
            f"- Median latency: {answers['median_seconds']}s (local 8B model on a laptop)",
        ]
    out = Path("docs") / "EVAL_RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
