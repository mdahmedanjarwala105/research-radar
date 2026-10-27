"""Trend radar: which research ideas are rising or fading?

Deliberately *not* done by the LLM. Asking an LLM "what's trending?" invites
confident guesses. Instead we compute evidence with plain statistics, then
let the LLM only narrate numbers we hand it:

1. Pick papers in the time window (optionally only those relevant to a topic,
   found with semantic search over abstracts).
2. Extract candidate key phrases (2-3 word n-grams) from titles + abstracts.
3. Split the window into an "early" and a "recent" half and compare the
   share of papers mentioning each phrase in each half.
4. Rank by lift = recent share - early share (with minimum support so one
   paper can't make a trend).
"""

import re
from collections import Counter
from datetime import UTC, datetime, timedelta

from radar.retrieval import Retriever
from radar.store import Store

STOP = set(
    """a an the of for to in on and or with without by from via as at is are be been being this that these
    those we our it its their they which while into over under using use used based new novel paper
    propose proposed proposes present presents show shows demonstrate approach approaches method methods
    framework frameworks model models results result task tasks study work data across can also such than
    more most both however two three first one well large language llm llms large-language-models via
    performance existing significantly achieve achieves state-of-the-art sota extensive experiments
    benchmark benchmarks effective effectively improve improves improved improving across different
    various each several make makes making not only between among within through including where how
    what when has have had do does may might could would should will via toward towards e.g i.e et al
    further while thus then them other many much via generation retrieval-augmented augmented
    rag ground grounds grounding dominant paradigm design choices
    https http github com org code available publicly release released outperform outperforms outperforming
    consistently substantially strong stronger strongest baseline baselines percentage points point
    all four five six statistically significant significantly average relative absolute up gains gain
    competitive comparable compared superior remains remain key while""".split()
)
TOKEN = re.compile(r"[a-z][a-z0-9\-]+")


def phrases(text: str) -> set[str]:
    toks = TOKEN.findall(text.lower())
    out = set()
    for n in (2, 3):
        for i in range(len(toks) - n + 1):
            gram = toks[i : i + n]
            if any(t in STOP or len(t) < 3 for t in gram):
                continue
            out.add(" ".join(gram))
    return out


def week_start(d: datetime) -> str:
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def trend_report(
    store: Store,
    retriever: Retriever | None = None,
    topic: str | None = None,
    weeks: int = 8,
    min_support: int = 3,
    top_n: int = 10,
    topic_margin: float = 0.10,
) -> dict:
    since = datetime.now(UTC) - timedelta(weeks=weeks)
    papers = store.papers(since=since)
    if topic and retriever:
        # Rank every abstract in the window by semantic similarity to the topic.
        # Cosine scores from small embedding models are compressed and shift per
        # topic (best match may be 0.68 for one topic, 0.83 for another), so a
        # fixed cutoff fails. Keep papers within `margin` of the best match instead.
        hits = retriever.dense(topic, limit=max(len(papers), 1), flt=retriever._filter(weeks * 7, "abstract"))
        if hits:
            cutoff = hits[0].score - topic_margin
            keep = [h for h in hits if h.score >= cutoff] or hits
            if len(keep) < 30:
                keep = hits[:30]
            relevant = {h.paper_id for h in keep}
            papers = [p for p in papers if p["id"] in relevant]
    if len(papers) < 6:
        return {"topic": topic, "weeks": weeks, "papers": len(papers), "error": "Not enough papers in window."}

    for p in papers:
        p["_dt"] = datetime.fromisoformat(p["published"])
        p["_phrases"] = phrases(p["title"] + ". " + p["abstract"])

    weekly = Counter(week_start(p["_dt"]) for p in papers)
    mid = since + (datetime.now(UTC) - since) / 2
    early = [p for p in papers if p["_dt"] < mid]
    recent = [p for p in papers if p["_dt"] >= mid]
    df_early = Counter(ph for p in early for ph in p["_phrases"])
    df_recent = Counter(ph for p in recent for ph in p["_phrases"])

    rows = []
    for ph in set(df_early) | set(df_recent):
        e, r = df_early[ph], df_recent[ph]
        if e + r < min_support:
            continue
        share_e = e / max(len(early), 1)
        share_r = r / max(len(recent), 1)
        rows.append({"phrase": ph, "early": e, "recent": r, "lift": round(share_r - share_e, 4)})

    rising = sorted((x for x in rows if x["recent"] >= min_support), key=lambda x: x["lift"], reverse=True)
    fading = sorted((x for x in rows if x["early"] >= min_support), key=lambda x: x["lift"])
    popular = sorted(rows, key=lambda x: x["early"] + x["recent"], reverse=True)

    top_rising = rising[:top_n]
    examples, used = [], set()
    for row in top_rising[:6]:
        match = next((p for p in reversed(recent) if row["phrase"] in p["_phrases"] and p["id"] not in used), None)
        if match:
            used.add(match["id"])
            examples.append({"phrase": row["phrase"], "paper_id": match["id"], "title": match["title"],
                             "published": match["published"][:10]})

    return {
        "topic": topic,
        "weeks": weeks,
        "papers": len(papers),
        "early_papers": len(early),
        "recent_papers": len(recent),
        "weekly_counts": dict(sorted(weekly.items())),
        "rising": top_rising,
        "fading": fading[:top_n],
        "popular": popular[:top_n],
        "examples": examples,
    }
