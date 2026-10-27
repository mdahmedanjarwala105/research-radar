"""Fetch AI papers from the public arXiv API (free, no key needed).

arXiv asks clients to wait ~3 seconds between requests, so paging is slow on
purpose. PDFs are cached on disk so re-running ingestion never re-downloads.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import feedparser
import httpx

from radar.config import settings

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100
POLITE_DELAY_S = 3.0


@dataclass
class Paper:
    id: str  # e.g. "2410.12345" (version suffix stripped)
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    categories: list[str]
    pdf_url: str
    has_fulltext: bool = False
    extra: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_entry(e) -> Paper:
    raw_id = e.id.rsplit("/abs/", 1)[-1]
    pdf = next((l.href for l in e.links if getattr(l, "title", "") == "pdf"), None)
    return Paper(
        id=re.sub(r"v\d+$", "", raw_id),
        title=_clean(e.title),
        abstract=_clean(e.summary),
        authors=[a.name for a in e.get("authors", [])],
        published=datetime.fromisoformat(e.published.replace("Z", "+00:00")),
        categories=[t.term for t in e.get("tags", [])],
        pdf_url=pdf or f"https://arxiv.org/pdf/{raw_id}",
    )


def build_query(topic: str | None, since: datetime | None, categories: list[str]) -> str:
    parts = ["(" + " OR ".join(f"cat:{c}" for c in categories) + ")"]
    if topic:
        # all: searches title + abstract + comments. Quote it as a phrase.
        parts.append(f'all:"{topic}"')
    if since:
        now = datetime.now(UTC)
        parts.append(f"submittedDate:[{since:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]")
    return " AND ".join(parts)


def fetch_papers(
    topic: str | None = None,
    max_results: int = 200,
    days: int | None = 60,
    since: datetime | None = None,
    categories: list[str] | None = None,
    on_page=None,
) -> list[Paper]:
    """Newest-first papers matching the topic, paging through the API."""
    if since is None and days:
        since = datetime.now(UTC) - timedelta(days=days)
    query = build_query(topic, since, categories or settings.arxiv_categories)
    papers: list[Paper] = []
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        start = 0
        while start < max_results:
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(PAGE_SIZE, max_results - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            feed = None
            for attempt in range(3):  # arXiv occasionally returns an empty page under load
                resp = http.get(API_URL, params=params)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                if feed.entries:
                    break
                time.sleep(POLITE_DELAY_S * (attempt + 1))
            if not feed or not feed.entries:
                break
            papers.extend(_parse_entry(e) for e in feed.entries)
            if on_page:
                on_page(len(papers))
            start += len(feed.entries)
            time.sleep(POLITE_DELAY_S)
    # The API can repeat entries across pages; keep the first of each id.
    seen, unique = set(), []
    for p in papers:
        if p.id not in seen:
            seen.add(p.id)
            unique.append(p)
    return unique


def download_pdf(paper: Paper, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or settings.pdf_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{paper.id.replace('/', '_')}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    with httpx.Client(timeout=120, follow_redirects=True) as http:
        resp = http.get(paper.pdf_url)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    time.sleep(POLITE_DELAY_S)
    return path
