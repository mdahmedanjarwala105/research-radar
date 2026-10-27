"""Storage layer: a vector database for chunks and SQLite for paper metadata.

Qdrant runs *embedded* here (a folder on disk, like SQLite), so there is no
server to manage. The same code works against a Qdrant server or Qdrant Cloud
by changing one line, which is how you'd scale it.

Each chunk is stored as one point carrying TWO vectors:
- "dense":  384 floats from an embedding model, capturing *meaning*
            ("LLM hallucination" ~ "model makes things up").
- "bm25":   a sparse vector of word ids -> weights, capturing *exact terms*
            (model names like "Qwen3", acronyms, numbers) that embeddings blur.
Keeping both lets retrieval run hybrid search over a single collection.
"""

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from functools import cached_property

from qdrant_client import QdrantClient, models

from radar.arxiv_source import Paper
from radar.chunking import Chunk
from radar.config import settings


class Models:
    """Loads the local ONNX models once, on first use (they take a few seconds)."""

    @cached_property
    def dense(self):
        from fastembed import TextEmbedding

        return TextEmbedding(settings.embed_model)

    @cached_property
    def sparse(self):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(settings.sparse_model)

    @cached_property
    def reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(settings.rerank_model)


class Store:
    def __init__(self):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.models = Models()
        self.qdrant = QdrantClient(path=str(settings.qdrant_path))
        self._ensure_collection()
        self.db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY, title TEXT, abstract TEXT, authors TEXT,
                published TEXT, categories TEXT, pdf_url TEXT,
                has_fulltext INTEGER DEFAULT 0, ingested_at TEXT
            )"""
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_published ON papers(published)")
        self.db.commit()

    def _ensure_collection(self):
        if self.qdrant.collection_exists(settings.collection):
            return
        self.qdrant.create_collection(
            settings.collection,
            vectors_config={
                "dense": models.VectorParams(size=settings.embed_dim, distance=models.Distance.COSINE)
            },
            # IDF modifier: Qdrant computes the "inverse document frequency" part
            # of BM25 itself, so rare words weigh more than common ones.
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in [
            ("paper_id", models.PayloadSchemaType.KEYWORD),
            ("kind", models.PayloadSchemaType.KEYWORD),
            ("published_ts", models.PayloadSchemaType.INTEGER),
        ]:
            self.qdrant.create_payload_index(settings.collection, field, schema)

    # ---------- papers (SQLite) ----------

    def upsert_paper(self, p: Paper):
        self.db.execute(
            """INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, abstract=excluded.abstract,
                 has_fulltext=MAX(papers.has_fulltext, excluded.has_fulltext)""",
            (
                p.id, p.title, p.abstract, json.dumps(p.authors), p.published.isoformat(),
                json.dumps(p.categories), p.pdf_url, int(p.has_fulltext),
            ),
        )
        self.db.commit()

    def has_paper(self, paper_id: str, need_fulltext: bool = False) -> bool:
        row = self.db.execute("SELECT has_fulltext FROM papers WHERE id=?", (paper_id,)).fetchone()
        return bool(row) and (not need_fulltext or bool(row["has_fulltext"]))

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        return _row(row) if row else None

    def papers(self, since: datetime | None = None) -> list[dict]:
        if since:
            rows = self.db.execute(
                "SELECT * FROM papers WHERE published >= ? ORDER BY published", (since.isoformat(),)
            )
        else:
            rows = self.db.execute("SELECT * FROM papers ORDER BY published")
        return [_row(r) for r in rows]

    def latest_published(self) -> datetime | None:
        row = self.db.execute("SELECT MAX(published) AS m FROM papers").fetchone()
        return datetime.fromisoformat(row["m"]) if row and row["m"] else None

    def stats(self) -> dict:
        r = self.db.execute(
            "SELECT COUNT(*) n, SUM(has_fulltext) f, MIN(published) a, MAX(published) b FROM papers"
        ).fetchone()
        return {
            "papers": r["n"],
            "papers_with_fulltext": r["f"] or 0,
            "chunks": self.qdrant.count(settings.collection).count,
            "oldest": r["a"],
            "newest": r["b"],
        }

    # ---------- chunks (Qdrant) ----------

    def index_chunks(self, chunks: list[Chunk], paper: Paper, batch_size: int = 32):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.indexed_text for c in batch]
            dense = list(self.models.dense.embed(texts))
            sparse = list(self.models.sparse.embed(texts))
            self.qdrant.upsert(
                settings.collection,
                points=[
                    models.PointStruct(
                        id=c.id,
                        vector={
                            "dense": d.tolist(),
                            "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                        },
                        payload={
                            "paper_id": c.paper_id,
                            "title": paper.title,
                            "section": c.section,
                            "kind": c.kind,
                            "position": c.position,
                            "text": c.text,
                            "published": paper.published.isoformat(),
                            "published_ts": int(paper.published.timestamp()),
                            "categories": paper.categories,
                        },
                    )
                    for c, d, s in zip(batch, dense, sparse)
                ],
            )

    def scroll_chunks(self, kind: str | None = None, limit: int = 10_000) -> Iterable[models.Record]:
        flt = (
            models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value=kind))])
            if kind
            else None
        )
        offset = None
        seen = 0
        while seen < limit:
            points, offset = self.qdrant.scroll(
                settings.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True
            )
            yield from points
            seen += len(points)
            if offset is None:
                break


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["authors"] = json.loads(d["authors"])
    d["categories"] = json.loads(d["categories"])
    d["has_fulltext"] = bool(d["has_fulltext"])
    return d
