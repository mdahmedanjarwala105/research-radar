"""The agentic RAG loop.

"Naive RAG" = retrieve top-k once, stuff into prompt, answer. It fails when
the question is vague, the first search misses, or the answer needs data
that isn't a text passage (e.g. "what's trending?").

This agent is an explicit state machine (the same idea frameworks like
LangGraph formalise), where the LLM makes small, schema-constrained
decisions between deterministic steps:

    PLAN ──► route = "direct"  ──────────────────────────────► ANSWER
      │      route = "trends"  ──► TREND STATS ──────────────► ANSWER
      │      route = "search"  ──► RETRIEVE ──► GRADE ──┬───► ANSWER ──► VERIFY
      │                               ▲                 │
      │                               └── REWRITE ◄─────┘ (evidence insufficient,
      │                                                    max N rounds)

Patterns used (all have papers behind them, good interview vocabulary):
- Query planning / multi-query rewriting   (RAG-Fusion)
- Routing to different tools               (Adaptive-RAG)
- Grading retrieved evidence + retrying    (Corrective RAG, "CRAG")
- Answer must cite sources inline          (attributed generation)
- Post-hoc claim verification              (Self-RAG style self-critique)

`run()` is a generator of events so the CLI, web UI and API can all show
the agent's reasoning live as it happens.
"""

import re
from collections.abc import Iterator
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from radar.config import settings
from radar.llm import LLM
from radar.retrieval import Hit, Retriever
from radar.store import Store
from radar.tracing import Trace
from radar.trends import trend_report


class Plan(BaseModel):
    route: Literal["search", "trends", "direct"] = Field(
        description="search = answer from paper text; trends = question about what is rising/popular/changing "
        "over time; direct = general definition or chit-chat needing no papers"
    )
    search_queries: list[str] = Field(
        description="1-3 short search queries phrased the way a paper would phrase it; include synonyms/acronyms"
    )
    recent_days: int | None = Field(
        description="If the user limits time (e.g. 'this month' = 30, 'this week' = 7), else null"
    )
    trend_topic: str | None = Field(description="For route=trends: the topic to analyse, or null for everything")


class Grade(BaseModel):
    relevant: list[int] = Field(description="Numbers of passages that help answer the question")
    sufficient: bool = Field(description="True if the relevant passages are enough for a good answer")
    missing: str = Field(description="What information is still missing; empty if sufficient")
    follow_up_query: str = Field(description="A new search query to find the missing info; empty if sufficient")


class Verification(BaseModel):
    supported: bool = Field(description="True if every factual claim is backed by the cited passages")
    unsupported_claims: list[str] = Field(description="Claims not backed by the passages (empty if none)")


PLAN_SYS = """You are the planner for a research assistant over a database of recent arXiv AI papers.
Today is {today}. Decide how to answer the user's question.

Routes:
- "trends": the user asks what is trending, rising, hot, popular, emerging, fading, or how the field
  is changing over time. Example: "What's trending in RAG?" / "Which topics are growing lately?"
- "search": the user asks about methods, findings, results, papers, comparisons or techniques.
  This is the default. Example: "How do papers reduce hallucination?" / "What did paper X find?"
- "direct": ONLY greetings or a pure definition of a basic term. Example: "What does RAG stand for?"

Rules:
- Never put years or dates inside search queries; the database only holds recent papers anyway.
- Set recent_days only for an explicit window ("this week" = 7, "last two months" = 60).
  Words like "recent", "latest" or "new" alone mean null.
Output JSON only."""

GRADE_SYS = """You judge whether retrieved passages from research papers can answer a question.
Be strict: a passage is relevant only if it contains information that directly helps answer.
Output JSON only."""

ANSWER_SYS = """You are Research Radar, an assistant that answers questions about recent AI research.
Use ONLY the numbered passages provided; never outside knowledge for factual claims."""

# Repeated at the END of the prompt: small models follow the most recent instructions best.
ANSWER_RULES = """Answer the question above using only the passages.
- Cite every sentence inline with passage numbers, e.g. "... improves recall [2][5]."
- Name the paper or method when you mention a finding.
- At most 6 bullets or 150 words. No headings, no emojis, no preamble, no closing offer.
- If the passages only partly answer the question, answer that part and say plainly what is missing."""

TREND_SYS = """You are Research Radar. You are given statistics computed from arXiv papers:
phrase counts in an early vs a recent time window, and example papers.
Write a short trend brief: what is rising, what is fading, what stays popular.
Only state what the numbers support; quote the counts. Cite example papers by their [n] number.
Be concise and concrete."""

DIRECT_SYS = """You are Research Radar, an assistant for AI research. Answer briefly from general
knowledge. End by suggesting one specific question the user could ask about recent papers."""

VERIFY_SYS = """You check an answer against its source passages. A claim is supported only if a cited
passage states it. Ignore style. Output JSON only."""


TIME_WORDS = re.compile(r"\b(today|yesterday|days?|weeks?|months?|years?|since|quarter)\b", re.I)


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.title} — {h.section}, arXiv:{h.paper_id}, {h.payload['published'][:10]})\n{h.text}"
        for i, h in enumerate(hits, 1)
    )


def source_list(hits: list[Hit]) -> list[dict]:
    return [
        {
            "n": i,
            "paper_id": h.paper_id,
            "title": h.title,
            "section": h.section,
            "published": h.payload["published"][:10],
            "url": f"https://arxiv.org/abs/{h.paper_id}",
            "snippet": h.text[:300],
            "found_by": sorted({r.split(":")[0] for r in h.ranks}),
            "rerank_score": round(h.rerank_score, 3) if h.rerank_score is not None else None,
        }
        for i, h in enumerate(hits, 1)
    ]


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}


def citation_coverage(answer: str) -> float:
    """Share of substantive sentences/bullets carrying at least one [n] citation."""
    units = [u for u in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(u.split()) >= 6]
    if not units:
        return 0.0
    return round(sum(bool(re.search(r"\[\d+", u)) for u in units) / len(units), 2)


class ResearchAgent:
    def __init__(self, store: Store, retriever: Retriever | None = None, llm: LLM | None = None):
        self.store = store
        self.retriever = retriever or Retriever(store)
        self.llm = llm or LLM()

    def run(self, question: str, verify: bool | None = None) -> Iterator[dict]:
        verify = settings.verify_answers if verify is None else verify
        trace = Trace(question)

        # ---- 1. PLAN ----
        with trace.span("plan") as s:
            plan = self.llm.structured(PLAN_SYS.format(today=date.today().isoformat()), f"Question: {question}", Plan)
            plan.search_queries = [q for q in plan.search_queries if q.strip()][:3] or [question]
            # Guardrail: the 8B planner sometimes invents a time window for words like
            # "recent". Only honour one if the user actually mentioned a time unit.
            if plan.recent_days and not TIME_WORDS.search(question):
                plan.recent_days = None
            s.update(plan.model_dump())
        yield {"type": "step", "name": "plan", "detail": plan.model_dump(), "ms": s["ms"]}

        if plan.route == "direct":
            yield from self._answer(trace, DIRECT_SYS, f"Question: {question}", hits=[], verify=False,
                                    note="General knowledge: no papers were consulted for this answer.")
            return
        if plan.route == "trends":
            yield from self._trends(trace, question, plan)
            return

        # ---- 2. RETRIEVE ⇄ GRADE (corrective loop) ----
        queries = [question, *plan.search_queries]
        evidence: dict[str, Hit] = {}
        for round_no in range(1, settings.max_retrieval_rounds + 1):
            with trace.span("retrieve", round=round_no, queries=queries) as s:
                hits = self.retriever.search(queries, since_days=plan.recent_days)
                s["n"] = len(hits)
            yield {
                "type": "step", "name": "retrieve", "ms": s["ms"],
                "detail": {"round": round_no, "queries": queries,
                           "top": [f"{h.title[:60]} ({h.section})" for h in hits[:5]]},
            }

            new = [h for h in hits if h.id not in evidence]
            with trace.span("grade", round=round_no) as s:
                grade = self.llm.structured(
                    GRADE_SYS,
                    f"Question: {question}\n\nPassages:\n{format_passages(new)}",
                    Grade,
                ) if new else Grade(relevant=[], sufficient=bool(evidence), missing="", follow_up_query="")
                kept = [new[i - 1] for i in grade.relevant if 1 <= i <= len(new)]
                for h in kept:
                    evidence[h.id] = h
                s.update(kept=len(kept), sufficient=grade.sufficient)
            yield {
                "type": "step", "name": "grade", "ms": s["ms"],
                "detail": {"round": round_no, "kept": len(kept), "of": len(new),
                           "sufficient": grade.sufficient, "missing": grade.missing},
            }
            if grade.sufficient or not grade.follow_up_query or round_no == settings.max_retrieval_rounds:
                break
            queries = [grade.follow_up_query]  # REWRITE and search again

        hits = list(evidence.values())[: settings.final_k]
        if not hits:
            msg = ("I couldn't find passages in the indexed papers that answer this. "
                   "Try rephrasing, or ingest more papers on this topic (`radar ingest --topic ...`).")
            trace.save(answer=msg, sources=[])
            yield {"type": "final", "answer": msg, "sources": [], "verification": None,
                   "trace_id": trace.id, "total_ms": trace.total_ms}
            return

        # ---- 3. ANSWER (+ 4. VERIFY) ----
        caveat = "" if grade.sufficient else f"\nNote: the evidence is incomplete. Missing: {grade.missing}\n"
        user = f"Question: {question}\n\nPassages:\n{format_passages(hits)}\n{caveat}\n{ANSWER_RULES}"
        yield from self._answer(trace, ANSWER_SYS, user, hits, verify)

    def _trends(self, trace: Trace, question: str, plan: Plan) -> Iterator[dict]:
        weeks = max(2, (plan.recent_days or 56) // 7)
        with trace.span("trend_stats", topic=plan.trend_topic, weeks=weeks) as s:
            report = trend_report(self.store, self.retriever, plan.trend_topic, weeks=weeks)
            s["papers"] = report.get("papers")
        yield {"type": "step", "name": "trend_stats", "ms": s["ms"], "detail": report}
        if "error" in report:
            yield {"type": "final", "answer": report["error"], "sources": [], "verification": None,
                   "trace_id": trace.id, "total_ms": trace.total_ms}
            return

        hits = []
        for ex in report["examples"]:
            p = self.store.get_paper(ex["paper_id"])
            hits.append(Hit(id=p["id"], payload={"paper_id": p["id"], "title": p["title"], "section": "Abstract",
                                                  "text": p["abstract"], "published": p["published"]}))
        fmt = lambda rows: "\n".join(f"- {r['phrase']}: early {r['early']} → recent {r['recent']}" for r in rows)
        user = (
            f"Question: {question}\n\nWindow: last {report['weeks']} weeks, {report['papers']} papers "
            f"({report['early_papers']} early half, {report['recent_papers']} recent half).\n"
            f"Papers per week: {report['weekly_counts']}\n\nRISING:\n{fmt(report['rising'])}\n\n"
            f"FADING:\n{fmt(report['fading'])}\n\nMOST COMMON:\n{fmt(report['popular'])}\n\n"
            f"Example papers:\n{format_passages(hits)}\n\n"
            "Write the brief now: at most 8 bullets, quote counts as 'early → recent', "
            "cite example papers as [n]. No headings, no emojis, no closing offer."
        )
        yield from self._answer(trace, TREND_SYS, user, hits, verify=False)

    def _answer(
        self, trace: Trace, system: str, user: str, hits: list[Hit], verify: bool, note: str | None = None
    ) -> Iterator[dict]:
        with trace.span("generate") as s:
            parts = []
            for tok in self.llm.stream(system, user, max_tokens=600):
                parts.append(tok)
                yield {"type": "token", "text": tok}
            answer = "".join(parts).strip()
        yield {"type": "step", "name": "generate", "ms": s["ms"], "detail": {"chars": len(answer)}}

        sources = source_list(hits)
        cited = cited_numbers(answer)
        bad_refs = sorted(n for n in cited if not 1 <= n <= len(hits))
        verification = None
        if verify and hits:
            with trace.span("verify") as s:
                v = self.llm.structured(
                    VERIFY_SYS, f"Passages:\n{format_passages(hits)}\n\nAnswer:\n{answer}", Verification
                )
                claims = [c for c in v.unsupported_claims if c.strip()]
                coverage = citation_coverage(answer)
                verification = {
                    # Don't trust the model's boolean alone: small models sometimes say
                    # "supported: true" while listing unsupported claims. Code decides.
                    "supported": v.supported and not claims and not bad_refs,
                    "unsupported_claims": claims,
                    "invalid_citations": bad_refs,
                    "citation_coverage": coverage,
                }
                s.update(verification)
            yield {"type": "step", "name": "verify", "ms": s["ms"], "detail": verification}

        for src in sources:
            src["cited"] = src["n"] in cited
        trace.save(answer=answer, sources=[s["paper_id"] for s in sources], verification=verification)
        yield {"type": "final", "answer": answer, "sources": sources, "verification": verification,
               "note": note, "trace_id": trace.id, "total_ms": trace.total_ms}

    def ask(self, question: str, verify: bool | None = None) -> dict:
        """Non-streaming convenience: run the loop and return the final event (+ steps)."""
        steps = []
        for ev in self.run(question, verify):
            if ev["type"] == "step":
                steps.append(ev)
            elif ev["type"] == "final":
                return {**ev, "steps": steps}
        raise RuntimeError("agent ended without a final answer")
