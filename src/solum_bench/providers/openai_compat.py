"""OpenAI-compatible model provider (GPT-4, OpenRouter, vLLM server, etc.)."""

from __future__ import annotations

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class OpenAICompatProvider:
    """Provider for OpenAI-compatible APIs."""

    provider_type = "openai"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install 'solum-bench[openai]'")

        self.name = name or model_id
        self.model_id = model_id
        self._api_base = api_base
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            client_kwargs: dict = {}
            if self._api_base:
                client_kwargs["base_url"] = self._api_base
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**client_kwargs)
        return self._client

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        kwargs: dict = {
            "model": self.model_id,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed

        with Timer() as timer:
            response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        thinking, clean_text = extract_thinking(raw_text)

        usage = response.usage
        return ModelResponse(
            text=clean_text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=timer.elapsed_ms,
            finish_reason=choice.finish_reason or "unknown",
            raw_response={"id": response.id, "model": response.model},
            thinking=thinking,
        )
