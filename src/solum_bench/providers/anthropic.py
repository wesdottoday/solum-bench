"""Anthropic (Claude) model provider."""

from __future__ import annotations

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class AnthropicProvider:
    """Provider for Claude models via the Anthropic API."""

    provider_type = "anthropic"

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install 'solum-bench[anthropic]'")

        self.name = name or model_id
        self.model_id = model_id
        self.client = anthropic.Anthropic()
        self.extra_kwargs = kwargs

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if system:
            kwargs["system"] = system

        with Timer() as timer:
            response = self.client.messages.create(**kwargs)

        raw_text = response.content[0].text
        thinking, clean_text = extract_thinking(raw_text)

        return ModelResponse(
            text=clean_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=timer.elapsed_ms,
            finish_reason=response.stop_reason or "unknown",
            raw_response={"id": response.id, "model": response.model},
            thinking=thinking,
        )
