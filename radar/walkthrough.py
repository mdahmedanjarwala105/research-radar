"""`radar walkthrough "question"`: run one question through the pipeline slowly,
stage by stage, showing the real numbers at every step and which code produced them.

This is a learning tool, not part of the product. Stages 1-6 call the retrieval
building blocks one at a time (single query, no LLM) so you can see each one's
effect in isolation. Stage 7 then runs the real agent, with a "spy" wrapped
around the LLM that prints every prompt it sends and every decision it gets back.
"""

import math

import numpy as np
from qdrant_client import models
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from radar.agent import ResearchAgent
from radar.config import settings
from radar.llm import LLM
from radar.retrieval import Retriever, limit_per_paper, rrf
from radar.store import Store

OFF_TOPIC = "A simple recipe for banana bread with walnuts and cinnamon."


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"… [{len(text) - n} more chars]"


class SpyLLM(LLM):
    """Same LLM, but prints every prompt and every structured answer.

    It works because the agent only talks to the LLM through these three
    methods, so wrapping them sees everything (this is "dependency injection":
    the agent receives its LLM instead of creating it).
    """

    def __init__(self, console: Console):
        super().__init__()
        self.console = console

    def _show_prompt(self, kind: str, system: str, user: str):
        approx_tokens = (len(system) + len(user)) // 4
        self.console.print(Panel(
            f"[bold]SYSTEM[/]\n{_short(system, 500)}\n\n[bold]USER[/]\n{_short(user, 1400)}",
            title=f"→ prompt sent to {self.model} ({kind}, ~{approx_tokens} tokens)",
            border_style="magenta",
        ))

    def structured(self, system, user, schema, temperature=0.0):
        self._show_prompt(f"structured: must return {schema.__name__} JSON", system, user)
        out = super().structured(system, user, schema, temperature)
        self.console.print(Panel(out.model_dump_json(indent=2), title=f"← {schema.__name__} returned",
                                 border_style="green"))
        return out

    def stream(self, system, user, temperature=0.2, max_tokens=None):
        self._show_prompt("free text, streamed", system, user)
        yield from super().stream(system, user, temperature, max_tokens)


STEP_NOTES = {
    "plan": ("The planner decided the route and wrote extra search queries.",
             "radar/agent.py → Plan, PLAN_SYS"),
    "retrieve": ("Hybrid search + RRF + rerank ran for EVERY query variant at once (multi-query).",
                 "radar/retrieval.py → Retriever.search()"),
    "grade": ("The LLM judged which passages actually help, and whether they are enough.",
              "radar/agent.py → Grade, the corrective loop in run()"),
    "trend_stats": ("Trend numbers computed by statistics, not by the LLM.", "radar/trends.py → trend_report()"),
    "generate": ("The answer was written using only the kept passages.", "radar/agent.py → _answer()"),
    "verify": ("A second LLM call checked each claim; code checked the \\[n] numbers.",
               "radar/agent.py → Verification, citation_coverage()"),
}


class Walkthrough:
    def __init__(self, question: str, pause: bool = True, console: Console | None = None):
        self.q = question
        self.pause = pause
        self.c = console or Console()
        self.store = Store()
        self.r = Retriever(self.store)
        self.m = self.store.models
        self.n_stage = 0

    def stage(self, title: str, what: str, code: str, try_this: str | None = None):
        if self.n_stage and self.pause:
            self.c.input("\n[dim]Press Enter for the next stage…[/]")
        self.n_stage += 1
        body = f"{what}\n\n[cyan]Code:[/] {code}"
        if try_this:
            body += f"\n[yellow]Try:[/] {try_this}"
        self.c.rule(f"[bold]Stage {self.n_stage} · {title}")
        self.c.print(body + "\n")

    def run(self):
        n = settings.candidates_per_retriever
        self.c.print(Panel(self.q, title="Question", border_style="bold"))
        dense = self.r.dense(self.q, n)
        sparse = self.r.sparse(self.q, n)

        # ---------------- 1. what's in the index ----------------
        top = dense[0]
        stats = self.store.stats()
        siblings = self.store.qdrant.count(settings.collection, count_filter=models.Filter(must=[
            models.FieldCondition(key="paper_id", match=models.MatchValue(value=top.paper_id))])).count
        self.stage(
            "What's in the index (built earlier by ingestion)",
            f"The index holds {stats['papers']} papers cut into {stats['chunks']} chunks. Below is the chunk "
            f"closest in meaning to your question (dense search #1). Its paper was cut into {siblings} chunks. "
            "The 'Paper/Section' header was added before embedding, so the vector knows the context.",
            "radar/ingest.py → ingest_papers() · radar/chunking.py → chunk_paper(), header()",
            "uv run radar search \"<your question>\" to see raw chunks for any query",
        )
        self.c.print(Panel(_short(top.full_text, 700), title=f"chunk {top.id[:8]}… · "
                           f"{len(top.text.split())} words · kind={top.payload['kind']}", border_style="blue"))

        # ---------------- 2. embeddings ----------------
        qv = next(iter(self.m.dense.query_embed(self.q)))
        others = list(self.m.dense.embed([top.full_text, OFF_TOPIC]))
        cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.stage(
            "Dense embedding: meaning → numbers",
            f"Your question became a vector of {len(qv)} numbers. Texts with similar meaning get vectors "
            "pointing in a similar direction; cosine similarity measures that (1.0 = identical direction).",
            "radar/store.py → Models.dense · radar/retrieval.py → Retriever.dense()",
            "rephrase your question with totally different words: the top chunk often stays the same",
        )
        self.c.print(f"first 8 of {len(qv)} numbers: [{', '.join(f'{x:+.3f}' for x in qv[:8])}, …]  "
                     f"(length of vector = {np.linalg.norm(qv):.3f}: normalised)")
        t = Table("compared with", "cosine similarity")
        t.add_row(f"top chunk: {top.title[:55]}", f"[green]{cos(qv, others[0]):.3f}[/]")
        t.add_row(f"off-topic: {OFF_TOPIC}", f"[red]{cos(qv, others[1]):.3f}[/]")
        self.c.print(t)
        self.c.print("[dim]Note: small models squeeze scores into a narrow band (~0.4–0.9), "
                     "so the *ranking* matters more than the raw value.[/]")

        # ---------------- 3. BM25 ----------------
        bm = self.m.sparse.model
        words = bm.tokenizer.tokenize(self.q)
        stems = bm._stem(words)  # also drops stop-words like "how", "do", "in"
        n_chunks = stats["chunks"]
        self.stage(
            "BM25: exact words, weighted by rarity",
            "BM25 throws away meaning and keeps words: stop-words are dropped, words are cut to their "
            "stem, and each stem becomes an id. A chunk scores higher for containing RARE stems from your "
            "question. Rarity = IDF (inverse document frequency), computed below from your index.",
            "radar/store.py → Models.sparse · radar/retrieval.py → Retriever.sparse()",
            "add an exact model or method name to your question and watch BM25's top results change",
        )
        t = Table("word", "stem", "chunks containing it", "IDF (rarity)")
        for w in words:
            stem = bm._stem([w])
            if not stem:
                t.add_row(w, "[dim]stop-word, dropped[/]", "", "")
                continue
            vec = models.SparseVector(indices=[bm.compute_token_id(stem[0])], values=[1.0])
            df = len(self.store.qdrant.query_points(settings.collection, query=vec, using="bm25",
                                                    limit=n_chunks, with_payload=False).points)
            idf = math.log((n_chunks - df + 0.5) / (df + 0.5) + 1)
            t.add_row(w, stem[0], f"{df} / {n_chunks}", f"{idf:.2f}")
        self.c.print(t)
        self.c.print(f"[dim]Query vector: {len(set(stems))} non-zero entries out of a huge vocabulary "
                     "→ that's why it's called a *sparse* vector.[/]")

        # ---------------- 4. dense vs BM25 results ----------------
        self.stage(
            "Two retrievers, two different top-5s",
            f"Each retriever returned its own top {n}. Notice where they agree and where they don't. "
            "Dense scores are cosine similarities; BM25 scores are unbounded sums. "
            "They are on different scales, so you can't simply add them.",
            "radar/retrieval.py → Retriever.dense(), Retriever.sparse()",
        )
        for name, hits in (("dense (meaning)", dense), ("BM25 (keywords)", sparse)):
            t = Table("rank", "score", "paper", "section", title=name)
            for i, h in enumerate(hits[:5], 1):
                t.add_row(str(i), f"{h.score:.3f}", h.title[:55], h.section[:22])
            self.c.print(t)
        overlap = {h.id for h in dense} & {h.id for h in sparse}
        self.c.print(f"Chunks found by BOTH retrievers in their top {n}: [bold]{len(overlap)}[/]")

        # ---------------- 5. RRF ----------------
        fused = rrf({"dense": dense, "bm25": sparse}, settings.rrf_k)
        k = settings.rrf_k
        self.stage(
            "Reciprocal Rank Fusion: merge by rank, not score",
            f"score = Σ 1/({k} + rank) over each list the chunk appears in. Being found by both "
            "retrievers adds two terms, so agreement wins.",
            "radar/retrieval.py → rrf()",
            "RADAR_RRF_K=5 uv run radar walkthrough \"...\" → a small k makes rank #1 dominate",
        )
        t = Table("fused rank", "dense rank", "BM25 rank", "math", "RRF score", "paper")
        for i, h in enumerate(fused[:8], 1):
            terms = [f"1/({k}+{h.ranks[x]})" for x in ("dense", "bm25") if x in h.ranks]
            t.add_row(str(i), str(h.ranks.get("dense", "—")), str(h.ranks.get("bm25", "—")),
                      " + ".join(terms), f"{h.score:.4f}", h.title[:40])
        self.c.print(t)

        # ---------------- 6. rerank ----------------
        rrf_rank = {h.id: i for i, h in enumerate(fused, 1)}
        reranked = self.r.rerank(self.q, fused[:n])
        final = limit_per_paper(reranked, 3)[: settings.final_k]
        self.stage(
            "Cross-encoder rerank: the careful second look",
            f"The top {n} fused chunks are re-scored by a model that reads question + chunk TOGETHER. "
            "It's slower but much better at judging relevance. Then we keep the best "
            f"{settings.final_k}, at most 3 per paper.",
            "radar/retrieval.py → Retriever.rerank(), limit_per_paper()",
            "uv run radar search \"...\" --mode hybrid vs --mode hybrid_rerank to compare",
        )
        t = Table("new rank", "was (RRF)", "move", "rerank score", "paper", "section")
        for i, h in enumerate(final, 1):
            was = rrf_rank[h.id]
            move = f"[green]↑{was - i}[/]" if was > i else (f"[red]↓{i - was}[/]" if was < i else "=")
            t.add_row(str(i), str(was), move, f"{h.rerank_score:.2f}", h.title[:40], h.section[:18])
        self.c.print(t)
        self.c.print("[dim]These 8 chunks are what the LLM would read. Everything so far took well under "
                     "a second and used no LLM at all.[/]")

        # ---------------- 7. the agent ----------------
        self.stage(
            "The agent runs for real (LLM prompts visible)",
            "Now the full agent runs. It repeats retrieval with several query variants and adds LLM "
            "decisions: plan → retrieve → grade (→ retry) → answer → verify. Magenta boxes are the exact "
            "prompts sent to the model; green boxes are the JSON it was forced to return.",
            "radar/agent.py → ResearchAgent.run()",
            "ask something NOT in the index and watch the grade step reject everything",
        )
        agent = ResearchAgent(self.store, self.r, llm=SpyLLM(self.c))
        streaming = False
        for ev in agent.run(self.q):
            if ev["type"] == "token":
                if not streaming:
                    self.c.print("[bold]Answer (streaming):[/]")
                    streaming = True
                self.c.print(ev["text"], end="", markup=False, highlight=False)
            elif ev["type"] == "step":
                if streaming:
                    self.c.print("\n")
                    streaming = False
                note, code = STEP_NOTES.get(ev["name"], ("", ""))
                self.c.print(f"[bold cyan]✔ {ev['name']}[/] [dim]{ev['ms']} ms[/] · {note} [dim]({code})[/]")
            elif ev["type"] == "final":
                self.c.print()
                for s in ev["sources"]:
                    mark = "[green]✓ cited[/]" if s.get("cited") else "[dim]· unused[/]"
                    self.c.print(f"  [{s['n']}] {mark} {s['title'][:60]} [dim]found by {s['found_by']}[/]")
                if ev.get("verification"):
                    self.c.print(f"verification: {ev['verification']}")
                self.c.print(f"\nTotal {ev['total_ms'] / 1000:.1f}s. The full trace (every step + timing) "
                             f"was appended to data/traces/ with id [bold]{ev['trace_id']}[/].")
