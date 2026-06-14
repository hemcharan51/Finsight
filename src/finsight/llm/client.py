"""LLM access layer.

Design intent from the doc: "Pydantic + instructor — typed, guaranteed-parse".
We implement the same contract directly on the Anthropic SDK (tool-use forced to a
single schema), which keeps the dependency surface small and gives us a clean seam
for a deterministic *mock* backend so the whole grid runs offline with no key.

Model routing (§07) lives here: callers ask for tier "small" or "large".
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from finsight.config import Settings, get_settings
from finsight.llm.mock import MockBackend

T = TypeVar("T", bound=BaseModel)

Tier = str  # "small" | "large"


class LLMResult(BaseModel):
    text: str
    tokens: int = 0
    model: str = ""


class LLM:
    """Tiered structured-output client with an offline mock fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._mock = MockBackend()
        self._client = None
        if self.settings.use_live_llm:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key or None)
            except Exception:  # pragma: no cover - defensive: fall back to mock
                self._client = None

    @property
    def live(self) -> bool:
        return self._client is not None

    def _model_for(self, tier: Tier) -> str:
        return self.settings.model_large if tier == "large" else self.settings.model_small

    # -- structured output -------------------------------------------------
    async def structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        tier: Tier = "small",
        system: str = "",
        context: dict | None = None,
    ) -> tuple[T, int]:
        """Return a validated `schema` instance plus tokens spent.

        `context` carries structured hints the mock backend uses to produce a
        deterministic, sensible answer when there is no live model.
        """
        if not self.live:
            obj, tokens = self._mock.structured(prompt, schema, context or {})
            return obj, tokens

        tool = {
            "name": "emit",
            "description": f"Emit a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        msg = await self._client.messages.create(
            model=self._model_for(tier),
            max_tokens=2048,
            system=system or "You are FinSight, a precise financial-document analyst.",
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": prompt}],
        )
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        payload = next(b.input for b in msg.content if b.type == "tool_use")
        return schema.model_validate(payload), tokens

    # -- free-form completion ---------------------------------------------
    async def complete(
        self, prompt: str, *, tier: Tier = "small", system: str = "", context: dict | None = None
    ) -> LLMResult:
        if not self.live:
            text, tokens = self._mock.complete(prompt, context or {})
            return LLMResult(text=text, tokens=tokens, model="mock")
        msg = await self._client.messages.create(
            model=self._model_for(tier),
            max_tokens=2048,
            system=system or "You are FinSight, a precise financial-document analyst.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(
            text=text,
            tokens=msg.usage.input_tokens + msg.usage.output_tokens,
            model=self._model_for(tier),
        )


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM
