"""Google (Gemini) model provider."""

from __future__ import annotations

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class GoogleProvider:
    """Provider for Gemini models via the Google GenAI SDK."""

    provider_type = "google"

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install 'solum-bench[google]'")

        self.name = name or model_id
        self.model_id = model_id
        self.client = genai.Client()

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Convert messages to Gemini content format
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        with Timer() as timer:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )

        raw_text = response.text or ""
        thinking, clean_text = extract_thinking(raw_text)

        usage_meta = response.usage_metadata
        return ModelResponse(
            text=clean_text,
            input_tokens=usage_meta.prompt_token_count if usage_meta else 0,
            output_tokens=usage_meta.candidates_token_count if usage_meta else 0,
            latency_ms=timer.elapsed_ms,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown",
            raw_response={},
            thinking=thinking,
        )
