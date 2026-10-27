"""Minimal observability: time every pipeline step and keep a record of it.

Production LLM apps log a "trace" per request (tools like Langfuse or
LangSmith do this) so you can see which step was slow or which retrieval led
to a bad answer. This is the same idea in ~40 lines, written to JSONL.
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from radar.config import settings


class Trace:
    def __init__(self, question: str):
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.spans: list[dict] = []
        self.started = time.perf_counter()

    @contextmanager
    def span(self, name: str, **meta):
        record = {"name": name, **meta}
        t0 = time.perf_counter()
        try:
            yield record  # callers can add result details to the record
        finally:
            record["ms"] = round((time.perf_counter() - t0) * 1000)
            self.spans.append(record)

    @property
    def total_ms(self) -> int:
        return round((time.perf_counter() - self.started) * 1000)

    def save(self, **extra) -> Path:
        out_dir = settings.data_dir / "traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
        row = {
            "trace_id": self.id,
            "time": datetime.now(UTC).isoformat(),
            "question": self.question,
            "total_ms": self.total_ms,
            "spans": self.spans,
            **extra,
        }
        with path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return path
