"""REST API + web UI.

`/api/ask` streams Server-Sent Events (SSE): the browser receives each agent
step and each answer token the moment it's produced, the same technique
ChatGPT-style UIs use to make slow models feel responsive.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from radar.agent import ResearchAgent
from radar.retrieval import MODES, Retriever
from radar.store import Store
from radar.trends import trend_report

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store()
    retriever = Retriever(store)
    state.update(store=store, retriever=retriever, agent=ResearchAgent(store, retriever))
    yield
    store.qdrant.close()


app = FastAPI(title="Research Radar", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    verify: bool = True


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    return state["store"].stats()


@app.get("/api/search")
def search(q: str, mode: str = "hybrid_rerank", k: int = 8):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    hits = state["retriever"].search(q, mode=mode, k=k)
    return [{"paper_id": h.paper_id, "title": h.title, "section": h.section, "text": h.text,
             "ranks": h.ranks, "rerank_score": h.rerank_score} for h in hits]


@app.get("/api/trends")
def trends(topic: str | None = None, weeks: int = 8):
    return trend_report(state["store"], state["retriever"], topic, weeks)


@app.post("/api/ask")
def ask(req: AskRequest):
    def events():
        try:
            for ev in state["agent"].run(req.question, verify=req.verify):
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:  # surface errors to the UI instead of a silent dead stream
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
