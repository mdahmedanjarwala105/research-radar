"""The "direct" route guardrail, tested with a fake LLM and a fake retriever.

No Ollama and no index needed: we force the planner to say "direct" and control
the reranker score, so we test the code's decision, not the model's mood.
"""

import pytest

from radar.agent import Grade, Plan, ResearchAgent, Verification
from radar.config import settings
from radar.retrieval import Hit


class FakeLLM:
    def structured(self, system, user, schema, temperature=0.0):
        if schema is Plan:  # the planner always (wrongly or rightly) says "answer from memory"
            return Plan(route="direct", search_queries=[], recent_days=None, trend_topic=None)
        if schema is Grade:
            return Grade(relevant=[1], sufficient=True, missing="", follow_up_query="")
        return Verification(supported=True, unsupported_claims=[])

    def stream(self, system, user, temperature=0.2, max_tokens=None):
        yield "answer [1]" if "Passages:" in user else "hello"


class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def search(self, queries, k=None, **kw):
        h = Hit(id="c1", payload={"paper_id": "2609.08267", "title": "EAEV paper", "section": "Abstract",
                                  "text": "We propose EAEV ...", "published": "2026-09-08"})
        h.rerank_score = self.score
        h.ranks = {"bm25:0": 1}
        return [h]


@pytest.fixture(autouse=True)
def tmp_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)


def run(score):
    agent = ResearchAgent(store=None, retriever=FakeRetriever(score), llm=FakeLLM())
    events = list(agent.run("EAEV"))
    return [e["name"] for e in events if e["type"] == "step"], events[-1]


def test_direct_is_overridden_when_the_papers_match():
    steps, final = run(score=3.13)  # the real EAEV score
    assert steps[:2] == ["plan", "route_check"]
    assert "retrieve" in steps  # it searched the papers instead
    assert final["sources"] and final["note"] is None


def test_direct_is_kept_for_small_talk():
    steps, final = run(score=-9.55)  # the real score for "Hello!"
    assert "retrieve" not in steps
    assert final["sources"] == [] and "General knowledge" in final["note"]
