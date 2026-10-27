"""Retrieval: find the chunks most likely to answer a question.

Pipeline (each stage can be switched off to measure what it adds; see
evaluation.py):

  query(s) ──► dense search  (top 30, by meaning) ──┐
           └─► BM25 search   (top 30, by keywords) ─┼─► RRF fusion ─► cross-encoder rerank ─► top k
  (repeat for each query variant: "multi-query")  ──┘

Why not just dense search?
  Embeddings are great at paraphrase but weak on rare exact tokens: model
  names ("Qwen3"), acronyms ("CRAG"), numbers. BM25 is the opposite.
  Combining them ("hybrid search") is now the default in production RAG.

Why Reciprocal Rank Fusion (RRF)?
  Dense scores (cosine, ~0.3-0.9) and BM25 scores (unbounded) live on
  different scales and can't be added. RRF ignores scores and uses ranks only:
      score(doc) = Σ over lists  1 / (k + rank_in_that_list)
  A doc ranked well by several retrievers floats to the top. k=60 is the
  constant from the original paper (Cormack et al., 2009) and damps the
  advantage of being #1 vs #2.

Why rerank?
  Dense/BM25 encode query and document *separately* (fast, can pre-compute
  millions). A cross-encoder reads query + chunk *together* through a
  transformer, so it understands their interaction much better, but it's too
  slow to run on everything. So: cheap retrieval for recall, expensive
  reranking on ~30 candidates for precision. Classic two-stage retrieval.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from qdrant_client import models

from radar.chunking import header
from radar.config import settings
from radar.store import Store

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    id: str
    payload: dict
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # which retriever found it, at what rank
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        return self.payload["text"]

    @property
    def paper_id(self) -> str:
        return self.payload["paper_id"]

    @property
    def title(self) -> str:
        return self.payload["title"]

    @property
    def section(self) -> str:
        return self.payload["section"]

    @property
    def full_text(self) -> str:
        return header(self.title, self.section) + self.text


def rrf(ranked_lists: dict[str, list[Hit]], k: int = 60) -> list[Hit]:
    fused: dict[str, Hit] = {}
    for name, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            h = fused.setdefault(hit.id, Hit(id=hit.id, payload=hit.payload))
            h.score += 1.0 / (k + rank)
            h.ranks[name] = rank
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


def limit_per_paper(hits: list[Hit], max_per_paper: int) -> list[Hit]:
    """Keep answers from leaning on one paper: diversity helps synthesis questions."""
    counts: dict[str, int] = {}
    out = []
    for h in hits:
        counts[h.paper_id] = counts.get(h.paper_id, 0) + 1
        if counts[h.paper_id] <= max_per_paper:
            out.append(h)
    return out


class Retriever:
    def __init__(self, store: Store):
        self.store = store
        self.m = store.models

    def _filter(self, since_days: int | None, kind: str | None) -> models.Filter | None:
        must = []
        if since_days:
            ts = int((datetime.now(UTC) - timedelta(days=since_days)).timestamp())
            must.append(models.FieldCondition(key="published_ts", range=models.Range(gte=ts)))
        if kind:
            must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        return models.Filter(must=must) if must else None

    def dense(self, query: str, limit: int, flt=None) -> list[Hit]:
        vec = next(iter(self.m.dense.query_embed(query))).tolist()
        res = self.store.qdrant.query_points(
            settings.collection, query=vec, using="dense", query_filter=flt, limit=limit
        )
        return [Hit(id=str(p.id), payload=p.payload, score=p.score) for p in res.points]

    def sparse(self, query: str, limit: int, flt=None) -> list[Hit]:
        sv = next(iter(self.m.sparse.query_embed(query)))
        res = self.store.qdrant.query_points(
            settings.collection,
            query=models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
            using="bm25",
            query_filter=flt,
            limit=limit,
        )
        return [Hit(id=str(p.id), payload=p.payload, score=p.score) for p in res.points]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = list(self.m.reranker.rerank(query, [h.full_text for h in hits]))
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)

    def search(
        self,
        queries: str | list[str],
        mode: str = "hybrid_rerank",
        k: int | None = None,
        since_days: int | None = None,
        kind: str | None = None,
        max_per_paper: int = 3,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        queries = [queries] if isinstance(queries, str) else queries
        k = k or settings.final_k
        n = settings.candidates_per_retriever
        flt = self._filter(since_days, kind)

        lists: dict[str, list[Hit]] = {}
        for i, q in enumerate(queries):
            if mode in ("dense", "hybrid", "hybrid_rerank"):
                lists[f"dense:{i}"] = self.dense(q, n, flt)
            if mode in ("sparse", "hybrid", "hybrid_rerank"):
                lists[f"bm25:{i}"] = self.sparse(q, n, flt)

        # Single list + single query: keep the retriever's own order. Otherwise fuse.
        candidates = next(iter(lists.values())) if len(lists) == 1 else rrf(lists, settings.rrf_k)
        if mode == "hybrid_rerank":
            # Rerank against the user's original wording (the first query).
            candidates = self.rerank(queries[0], candidates[: n])
        return limit_per_paper(candidates, max_per_paper)[:k]
