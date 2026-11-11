"""All tunable knobs in one place.

Every value can be overridden with an environment variable prefixed RADAR_,
e.g. `RADAR_LLM_MODEL=llama3.1:8b uv run radar ask "..."`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RADAR_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # --- Models (all run locally, no API keys) ---
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    # Dense embedding model: turns text into a 384-number vector capturing meaning.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # Sparse model: classic BM25 keyword scoring, stored as a sparse vector.
    sparse_model: str = "Qdrant/bm25"
    # Cross-encoder: reads (query, chunk) together and scores relevance precisely.
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Data source ---
    arxiv_categories: list[str] = ["cs.CL", "cs.AI", "cs.LG", "cs.IR"]

    # --- Chunking ---
    chunk_words: int = 180
    chunk_overlap_sentences: int = 2

    # --- Retrieval ---
    collection: str = "papers"
    candidates_per_retriever: int = 30  # how many each retriever returns before fusion
    final_k: int = 8  # chunks handed to the LLM
    rrf_k: int = 60  # constant in Reciprocal Rank Fusion

    # --- Agent ---
    max_retrieval_rounds: int = 2  # corrective retries when context is insufficient
    verify_answers: bool = True
    # Guardrail on the "direct" route: if the reranker scores any passage above this,
    # the papers know something about the question, so search them instead of
    # answering from memory. Small talk scores around -7 to -10; real topics +3 or more.
    direct_route_max_rerank: float = 0.0

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "papers.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"


settings = Settings()
