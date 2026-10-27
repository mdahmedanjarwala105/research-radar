"""Command-line interface: `uv run radar --help`."""

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Research Radar: agentic RAG over the latest AI papers.", no_args_is_help=True)
eval_app = typer.Typer(help="Measure retrieval and answer quality.")
app.add_typer(eval_app, name="eval")
console = Console()


def _store():
    from radar.store import Store

    return Store()


@app.command()
def ingest(
    topic: str = typer.Option(None, help='arXiv phrase search, e.g. "retrieval augmented generation"'),
    max_papers: int = typer.Option(200, help="Max papers to fetch (abstracts are always indexed)"),
    fulltext: int = typer.Option(30, help="How many of the newest papers to also index in full text"),
    days: int = typer.Option(90, help="Only papers from the last N days"),
):
    """Fetch papers from arXiv and index them."""
    from radar.ingest import ingest as run

    report = run(_store(), topic, max_papers, fulltext, days, log=console.print)
    console.print(report)


@app.command()
def refresh(
    topic: str = typer.Option(None),
    max_papers: int = typer.Option(200),
    fulltext: int = typer.Option(10),
):
    """Incrementally add papers newer than the newest one stored (run this on a schedule)."""
    from radar.ingest import refresh as run

    console.print(run(_store(), topic, max_papers, fulltext, log=console.print))


@app.command()
def stats():
    """Show what's in the index."""
    console.print_json(json.dumps(_store().stats()))


@app.command()
def search(
    query: str,
    mode: str = typer.Option("hybrid_rerank", help="dense | sparse | hybrid | hybrid_rerank"),
    k: int = 8,
):
    """Raw retrieval, no LLM: see exactly which chunks come back and why."""
    from radar.retrieval import Retriever

    hits = Retriever(_store()).search(query, mode=mode, k=k)
    t = Table("#", "paper", "section", "found by (rank)", "rerank", "snippet", show_lines=True)
    for i, h in enumerate(hits, 1):
        found = ", ".join(f"{name}#{r}" for name, r in h.ranks.items()) or mode
        t.add_row(str(i), f"{h.title[:45]}\n[dim]{h.paper_id}[/]", h.section[:20], found,
                  f"{h.rerank_score:.2f}" if h.rerank_score is not None else "-", h.text[:160] + "…")
    console.print(t)


@app.command()
def ask(question: str, verify: bool = typer.Option(True, help="Run the claim-verification step")):
    """Ask the agent a question; watch it plan, retrieve, grade and answer."""
    from radar.agent import ResearchAgent

    agent = ResearchAgent(_store())
    mid_answer = False  # currently printing streamed tokens
    any_tokens = False
    for ev in agent.run(question, verify=verify):
        if ev["type"] == "step":
            if mid_answer:
                console.print("\n")
                mid_answer = False
            d = ev["detail"]
            if ev["name"] == "trend_stats":
                d = {k: d.get(k) for k in ("topic", "weeks", "papers", "rising")}
                d["rising"] = [r["phrase"] for r in d.get("rising") or []][:6]
            console.print(f"[cyan]▸ {ev['name']}[/] [dim]({ev['ms']} ms)[/] {json.dumps(d, default=str)[:300]}")
        elif ev["type"] == "token":
            if not any_tokens:
                console.print("\n[bold]Answer[/]")
            any_tokens = mid_answer = True
            console.print(ev["text"], end="", markup=False, highlight=False)
        elif ev["type"] == "final":
            if not any_tokens:  # e.g. abstained before generating
                console.print(f"\n[bold]Answer[/]\n{ev['answer']}")
            if ev.get("note"):
                console.print(f"[yellow]ⓘ {ev['note']}[/]")
            console.print()
            for s in ev["sources"]:
                mark = "[green]✓[/]" if s.get("cited") else "[dim]·[/]"
                console.print(f" {mark} [{s['n']}] {s['title'][:70]} [dim]({s['section'][:25]}) {s['url']}[/]")
            v = ev.get("verification")
            if v:
                ok = v["supported"] and not v["invalid_citations"]
                console.print(f"\n[{'green' if ok else 'yellow'}]verification:[/] {json.dumps(v)}")
            console.print(f"[dim]trace {ev['trace_id']} · {ev['total_ms'] / 1000:.1f}s[/]")


@app.command()
def trends(topic: str = typer.Option(None), weeks: int = 8):
    """Statistical trend radar (no LLM): rising and fading research phrases."""
    from radar.retrieval import Retriever
    from radar.trends import trend_report

    store = _store()
    r = trend_report(store, Retriever(store), topic, weeks)
    if "error" in r:
        console.print(f"[red]{r['error']}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{r['papers']} papers over {weeks} weeks[/]  weekly: {r['weekly_counts']}")
    for name in ("rising", "fading", "popular"):
        t = Table("phrase", "early", "recent", "lift", title=name.upper())
        for row in r[name]:
            t.add_row(row["phrase"], str(row["early"]), str(row["recent"]), f"{row['lift']:+.3f}")
        console.print(t)


@eval_app.command("build")
def eval_build(
    n: int = 40,
    style: str = typer.Option("specific", help="specific (names methods) | conceptual (plain words, no names)"),
):
    """Generate a synthetic question set from indexed chunks."""
    from radar.evaluation import build_testset
    from radar.llm import LLM

    console.print(f"Wrote {build_testset(_store(), LLM(), n=n, style=style, log=console.print)}")


@eval_app.command("run")
def eval_run(k: int = 8, answers: int = typer.Option(8, help="Questions to run end-to-end (0 to skip)")):
    """Compare retrieval modes on every test set, and score end-to-end answers."""
    from radar.agent import ResearchAgent
    from radar.evaluation import answer_eval, available_styles, load_testset, retrieval_eval, save_report
    from radar.retrieval import Retriever

    store = _store()
    retriever = Retriever(store)
    styles = available_styles()
    if not styles:
        console.print("[red]No test sets. Run `radar eval build` first.[/]")
        raise typer.Exit(1)
    by_style, sizes = {}, {}
    for style in styles:
        testset = load_testset(style)
        sizes[style] = len(testset)
        ret = by_style[style] = retrieval_eval(retriever, testset, k=k)
        t = Table("mode", *next(iter(ret.values())).keys(), title=f"Retrieval ablation — {style} questions")
        for mode, row in ret.items():
            t.add_row(mode, *map(str, row.values()))
        console.print(t)
    ans = None
    if answers:
        ans = answer_eval(ResearchAgent(store, retriever), load_testset(styles[0]), n=answers, log=console.print)
        console.print({k: v for k, v in ans.items() if k != "rows"})
    console.print(f"Report: {save_report(by_style, ans, sizes, k, store.stats())}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Web UI + REST API at http://127.0.0.1:8000"""
    import uvicorn

    uvicorn.run("radar.api:app", host=host, port=port)


@app.command()
def mcp():
    """Run as an MCP server over stdio (for Claude Desktop, Claude Code, Cursor...)."""
    from radar.mcp_server import server

    server.run("stdio")


if __name__ == "__main__":
    app()
