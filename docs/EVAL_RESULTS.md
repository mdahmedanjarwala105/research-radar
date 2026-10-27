# Evaluation results

_Generated 2026-10-27 15:21 UTC on 400 papers / 2584 chunks, k=8. Models: `BAAI/bge-small-en-v1.5`, `Qdrant/bm25`, `Xenova/ms-marco-MiniLM-L-6-v2`, `qwen3:8b`._

## Retrieval ablation — specific questions (n=40)

_Questions name the method/dataset (a reader who knows what they want). Exact names favour BM25 keyword search._

| mode | hit@1 | hit@3 | hit@8 | mrr | paper_hit@8 | p50_ms |
|---|---|---|---|---|---|---|
| dense | 0.5 | 0.675 | 0.875 | 0.624 | 0.95 | 6 |
| sparse | 0.775 | 0.925 | 0.975 | 0.858 | 1.0 | 35 |
| hybrid | 0.65 | 0.95 | 0.95 | 0.787 | 1.0 | 43 |
| hybrid_rerank | 0.7 | 0.975 | 0.975 | 0.821 | 1.0 | 780 |

## Retrieval ablation — conceptual questions (n=40)

_Questions describe the idea in plain words with no names or acronyms (a user who doesn't know the paper's vocabulary). This tests semantic retrieval._

| mode | hit@1 | hit@3 | hit@8 | mrr | paper_hit@8 | p50_ms |
|---|---|---|---|---|---|---|
| dense | 0.125 | 0.25 | 0.4 | 0.203 | 0.8 | 5 |
| sparse | 0.075 | 0.175 | 0.275 | 0.137 | 0.625 | 37 |
| hybrid | 0.1 | 0.275 | 0.525 | 0.217 | 0.8 | 42 |
| hybrid_rerank | 0.125 | 0.325 | 0.525 | 0.25 | 0.8 | 706 |

## End-to-end answers

- Questions: 8
- Faithfulness (verifier says every claim is supported): 88%
- Cites the gold paper: 100%
- Abstained (no evidence found): 0%
- Median latency: 19.65s (local 8B model on a laptop)
