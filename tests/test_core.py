"""Fast unit tests for the pure logic (no models, no network)."""

from radar.agent import cited_numbers
from radar.chunking import chunk_id, chunk_paper, pack_sentences, split_sentences
from radar.parsing import _heading
from radar.retrieval import Hit, limit_per_paper, rrf
from radar.trends import phrases


def hit(id, paper="p1"):
    return Hit(id=id, payload={"paper_id": paper, "text": id, "title": "T", "section": "S"})


def test_rrf_rewards_agreement_between_retrievers():
    dense = [hit("a"), hit("b"), hit("c")]
    bm25 = [hit("b"), hit("d"), hit("e")]
    fused = rrf({"dense": dense, "bm25": bm25}, k=60)
    # "b" is found by both retrievers (#2 and #1), so it beats "a" (#1 in one list only):
    # 1/62 + 1/61 > 1/61.
    assert fused[0].id == "b"
    assert fused[0].ranks == {"dense": 2, "bm25": 1}
    assert {h.id for h in fused} == {"a", "b", "c", "d", "e"}


def test_rrf_score_formula():
    fused = rrf({"x": [hit("a")]}, k=60)
    assert abs(fused[0].score - 1 / 61) < 1e-12


def test_limit_per_paper():
    hits = [hit("1", "A"), hit("2", "A"), hit("3", "A"), hit("4", "B")]
    assert [h.id for h in limit_per_paper(hits, 2)] == ["1", "2", "4"]


def test_sentence_split_and_pack_with_overlap():
    text = "First sentence here. Second one is here. Third is the last one."
    sents = split_sentences(text)
    assert len(sents) == 3
    chunks = pack_sentences(sents, max_words=7, overlap=1)
    assert len(chunks) >= 2
    # overlap: the last sentence of a chunk starts the next chunk
    assert chunks[1].startswith(sents[1]) or chunks[1].startswith(sents[0])


def test_chunk_paper_adds_contextual_header_and_stable_ids():
    chunks = chunk_paper("2401.0001", "Great Paper", "An abstract.", [("1 Introduction", "Hello world. " * 50)])
    assert chunks[0].kind == "abstract"
    assert chunks[1].indexed_text.startswith("Paper: Great Paper\nSection: 1 Introduction")
    assert chunks[1].id == chunk_id("2401.0001", "fulltext", 1)  # deterministic -> idempotent re-ingest


def test_heading_detection():
    assert _heading("3.2 Retrieval Module") == "3.2 Retrieval Module"
    assert _heading("References") == "References"
    assert _heading("Conclusion") == "Conclusion"
    assert _heading("We show that 2 models outperform the baseline.") is None


def test_cited_numbers():
    assert cited_numbers("A [1]. B [2, 3]. C [3][4].") == {1, 2, 3, 4}
    assert cited_numbers("no citations") == set()


def test_phrases_drop_stopwords():
    ps = phrases("We propose graph-based retrieval for multi-hop question answering.")
    assert "multi-hop question" in ps
    assert not any(p.startswith("we ") for p in ps)
