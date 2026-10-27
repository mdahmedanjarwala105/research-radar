"""Turn a research-paper PDF into (section heading, text) pairs.

PDFs store positioned glyphs, not paragraphs, so extraction is always a bit
messy. We keep it simple and explainable:
1. Extract text page by page with PyMuPDF.
2. Re-join words hyphenated across line breaks.
3. Detect section headings with a regex for numbered titles ("3.2 Method")
   and well-known unnumbered ones ("Abstract", "Conclusion").
4. Stop at "References": citation lists are noise for retrieval.
"""

import re
from pathlib import Path

import pymupdf

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method", "methods",
    "methodology", "approach", "experiments", "experimental setup", "results",
    "evaluation", "discussion", "limitations", "conclusion", "conclusions",
    "future work", "acknowledgments", "acknowledgements",
}
STOP_HEADINGS = {"references", "bibliography"}
NUMBERED = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z0-9 ,:&()\-/]{2,70})$")


def _heading(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if s.lower().rstrip(".") in KNOWN_HEADINGS | STOP_HEADINGS:
        return s.rstrip(".").title()
    m = NUMBERED.match(s)
    if m and not s.endswith(".") and sum(c.isdigit() for c in m.group(3)) < 4:
        return f"{m.group(1)} {m.group(3)}"
    return None


def extract_sections(pdf_path: Path, max_pages: int = 25) -> list[tuple[str, str]]:
    with pymupdf.open(pdf_path) as doc:
        raw = "\n".join(page.get_text() for page in doc.pages(0, min(max_pages, doc.page_count)))
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # "retrie-\nval" -> "retrieval"

    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    for line in raw.splitlines():
        h = _heading(line)
        if h:
            if h.lower() in STOP_HEADINGS:
                break
            sections.append((h, []))
        else:
            sections[-1][1].append(line)

    out = []
    for title, lines in sections:
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if len(text.split()) >= 30:  # skip empty or tiny fragments (figure labels etc.)
            out.append((title, text))
    return out
