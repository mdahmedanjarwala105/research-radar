"""Ingestion pipeline: arXiv -> (PDF -> sections) -> chunks -> vectors.

Designed to be run repeatedly (e.g. nightly via cron):
- Idempotent: chunk ids are deterministic, so re-runs overwrite, never duplicate.
- Incremental: `refresh` only asks arXiv for papers newer than the newest one
  already stored, so the index keeps up with the field cheaply.
- Graceful: one broken PDF is logged and skipped; the paper still gets its
  abstract indexed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from radar.arxiv_source import Paper, download_pdf, fetch_papers
from radar.chunking import chunk_paper
from radar.parsing import extract_sections
from radar.store import Store


@dataclass
class IngestReport:
    fetched: int = 0
    new_papers: int = 0
    fulltext_papers: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)


def ingest_papers(
    store: Store,
    papers: list[Paper],
    fulltext_limit: int,
    log: Callable[[str], None] = print,
) -> IngestReport:
    report = IngestReport(fetched=len(papers))
    fulltext_budget = fulltext_limit
    for i, paper in enumerate(papers, 1):
        want_fulltext = fulltext_budget > 0
        if store.has_paper(paper.id, need_fulltext=want_fulltext):
            continue
        report.new_papers += 1
        sections = None
        if want_fulltext:
            try:
                sections = extract_sections(download_pdf(paper))
                paper.has_fulltext = bool(sections)
            except Exception as e:  # network hiccup, malformed PDF, ...
                report.errors.append(f"{paper.id}: {type(e).__name__}: {e}")
        chunks = chunk_paper(paper.id, paper.title, paper.abstract, sections)
        store.index_chunks(chunks, paper)
        store.upsert_paper(paper)
        report.chunks += len(chunks)
        if paper.has_fulltext:
            report.fulltext_papers += 1
            fulltext_budget -= 1
        log(f"[{i}/{len(papers)}] {paper.id} {len(chunks):>3} chunks  {paper.title[:70]}")
    return report


def ingest(
    store: Store,
    topic: str | None,
    max_papers: int,
    fulltext: int,
    days: int,
    log: Callable[[str], None] = print,
) -> IngestReport:
    log(f"Fetching up to {max_papers} papers from arXiv (topic={topic!r}, last {days} days)...")
    papers = fetch_papers(topic=topic, max_results=max_papers, days=days, on_page=lambda n: log(f"  ...{n} fetched"))
    return ingest_papers(store, papers, fulltext_limit=fulltext, log=log)


def refresh(
    store: Store, topic: str | None, max_papers: int, fulltext: int, log: Callable[[str], None] = print
) -> IngestReport:
    latest = store.latest_published()
    since = latest - timedelta(hours=1) if latest else None  # small overlap; duplicates are skipped
    log(f"Refreshing papers published since {since or 'the beginning'}...")
    papers = fetch_papers(topic=topic, max_results=max_papers, days=None if since else 14, since=since)
    return ingest_papers(store, papers, fulltext_limit=fulltext, log=log)
