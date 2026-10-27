"""Split documents into retrieval-sized chunks.

Why chunk at all? Embedding a whole 20-page paper into one vector blurs every
idea in it together, and LLM context is limited. Small chunks let retrieval
pinpoint the exact passage that answers a question.

Strategy used here ("structure-aware chunking"):
- Never cross a section boundary, so a chunk is about one thing.
- Split on sentence boundaries, never mid-sentence.
- Pack sentences up to ~chunk_words, then start a new chunk that repeats the
  last couple of sentences (overlap), so an idea split across two chunks
  still appears whole in one of them.
- Prefix each chunk with a "contextual header" (paper title + section). A
  chunk saying "our method improves recall by 12%" is useless alone; with the
  header, both the embedding and the LLM know *whose* method it is. This is a
  cheap version of Anthropic's "Contextual Retrieval" technique.
"""

import re
import uuid
from dataclasses import dataclass

from radar.config import settings

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
NAMESPACE = uuid.UUID("6f1c1c1e-6a8e-4b8e-9a52-6b8b3a0e2d11")


@dataclass
class Chunk:
    id: str
    paper_id: str
    section: str
    kind: str  # "abstract" or "fulltext"
    position: int
    text: str  # the raw passage (shown to users / the LLM)
    indexed_text: str  # header + passage (what gets embedded)


def chunk_id(paper_id: str, kind: str, position: int) -> str:
    # Deterministic ids: re-ingesting the same paper overwrites instead of duplicating.
    return str(uuid.uuid5(NAMESPACE, f"{paper_id}:{kind}:{position}"))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def pack_sentences(
    sentences: list[str], max_words: int, overlap: int
) -> list[str]:
    chunks, current, words = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if current and words + n > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            words = sum(len(s.split()) for s in current)
        current.append(sent)
        words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def header(title: str, section: str) -> str:
    return f"Paper: {title}\nSection: {section}\n\n"


def chunk_paper(
    paper_id: str,
    title: str,
    abstract: str,
    sections: list[tuple[str, str]] | None = None,
    max_words: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    max_words = max_words or settings.chunk_words
    overlap = settings.chunk_overlap_sentences if overlap is None else overlap

    # The abstract is always its own chunk: it's the densest summary of the paper.
    chunks = [
        Chunk(
            id=chunk_id(paper_id, "abstract", 0),
            paper_id=paper_id,
            section="Abstract",
            kind="abstract",
            position=0,
            text=abstract,
            indexed_text=header(title, "Abstract") + abstract,
        )
    ]
    pos = 0
    for section, text in sections or []:
        if section.lower() == "abstract":
            continue
        for piece in pack_sentences(split_sentences(text), max_words, overlap):
            pos += 1
            chunks.append(
                Chunk(
                    id=chunk_id(paper_id, "fulltext", pos),
                    paper_id=paper_id,
                    section=section,
                    kind="fulltext",
                    position=pos,
                    text=piece,
                    indexed_text=header(title, section) + piece,
                )
            )
    return chunks
