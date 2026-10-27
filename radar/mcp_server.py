"""MCP server: lets any AI assistant use Research Radar as a tool.

The Model Context Protocol (MCP) is the open standard (from Anthropic, now
supported by OpenAI, Google, Microsoft, Cursor, ...) for plugging tools and
data into AI assistants. Instead of building one integration per assistant,
you expose an MCP server once and every MCP client can call it.

Here the assistant gets four tools. Note the design choice: `search_papers`
returns raw passages so a *strong* remote model (e.g. Claude) can do its own
reasoning, while `ask_research_question` runs our local agent end to end.

Register with Claude Code:
    claude mcp add research-radar -- uv --directory /path/to/RAG_AI run radar mcp
"""

from functools import cache

from mcp.server.mcpserver import MCPServer

from radar.agent import ResearchAgent
from radar.retrieval import Retriever
from radar.store import Store
from radar.trends import trend_report

server = MCPServer(
    "research-radar",
    instructions="Search and analyse recent arXiv AI papers (RAG, agents, LLMs). "
    "Prefer search_papers for evidence; cite arXiv ids from results.",
)


@cache
def _components():
    store = Store()
    retriever = Retriever(store)
    return store, retriever, ResearchAgent(store, retriever)


@server.tool()
def search_papers(query: str, k: int = 6, recent_days: int | None = None) -> list[dict]:
    """Hybrid search (dense + BM25 + rerank) over recent AI papers. Returns passages with arXiv ids."""
    _, retriever, _ = _components()
    return [
        {"arxiv_id": h.paper_id, "title": h.title, "section": h.section,
         "published": h.payload["published"][:10], "text": h.text}
        for h in retriever.search(query, k=k, since_days=recent_days)
    ]


@server.tool()
def ask_research_question(question: str) -> dict:
    """Run the local agent: plan, retrieve, grade, answer with citations, verify claims."""
    _, _, agent = _components()
    out = agent.ask(question)
    return {"answer": out["answer"], "verification": out["verification"],
            "sources": [{k: s[k] for k in ("n", "title", "url", "cited")} for s in out["sources"]]}


@server.tool()
def research_trends(topic: str | None = None, weeks: int = 8) -> dict:
    """Rising / fading research phrases computed from paper counts over time (no LLM guessing)."""
    store, retriever, _ = _components()
    r = trend_report(store, retriever, topic, weeks)
    return {k: r.get(k) for k in ("topic", "weeks", "papers", "weekly_counts", "rising", "fading", "examples", "error")}


@server.tool()
def get_paper(arxiv_id: str) -> dict:
    """Metadata and abstract for one indexed paper."""
    store, _, _ = _components()
    return store.get_paper(arxiv_id) or {"error": f"{arxiv_id} is not in the index"}
