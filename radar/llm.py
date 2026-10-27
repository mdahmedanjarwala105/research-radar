"""Thin wrapper around a local LLM served by Ollama.

Two kinds of calls:
- `chat`/`stream`: free-form text (used for the final answer).
- `structured`: the model must return JSON matching a Pydantic schema. Ollama
  enforces this with constrained decoding (the model literally cannot emit
  tokens that break the schema), which makes small 8B models reliable enough
  to drive an agent's decisions.

Everything else in the project talks to this class only, so swapping in a
different provider means changing this one file.
"""

from collections.abc import Iterator
from typing import TypeVar

import ollama
from pydantic import BaseModel

from radar.config import settings

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.llm_model
        self.client = ollama.Client(host=host or settings.ollama_host)

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        # think=False turns off qwen3's hidden reasoning tokens: much faster, and
        # our pipeline already breaks the problem into explicit steps.
        r = self.client.chat(
            model=self.model,
            messages=self._messages(system, user),
            think=False,
            options={"temperature": temperature},
        )
        return r.message.content.strip()

    def stream(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int | None = None
    ) -> Iterator[str]:
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens  # hard cap: protects latency if the model rambles
        for part in self.client.chat(
            model=self.model,
            messages=self._messages(system, user),
            think=False,
            stream=True,
            options=options,
        ):
            if part.message.content:
                yield part.message.content

    def structured(self, system: str, user: str, schema: type[T], temperature: float = 0.0) -> T:
        r = self.client.chat(
            model=self.model,
            messages=self._messages(system, user),
            think=False,
            format=schema.model_json_schema(),
            options={"temperature": temperature},
        )
        return schema.model_validate_json(r.message.content)
