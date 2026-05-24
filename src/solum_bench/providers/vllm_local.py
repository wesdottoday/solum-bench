"""vLLM offline inference provider for local GPU models."""

from __future__ import annotations

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class VLLMLocalProvider:
    """Provider for local models via vLLM offline inference."""

    provider_type = "vllm_local"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        dtype: str = "auto",
        tensor_parallel: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        **kwargs,
    ):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("Install vllm: pip install 'solum-bench[vllm]'")

        self.name = name or model_id
        self.model_id = model_id

        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        from vllm import SamplingParams

        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.extend({"role": m["role"], "content": m["content"]} for m in messages)

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # vLLM requires >0
            top_p=0.9 if temperature > 0 else 1.0,
            seed=seed,
        )

        with Timer() as timer:
            outputs = self.llm.chat([conversation], params)

        output = outputs[0]
        raw_text = output.outputs[0].text
        thinking, clean_text = extract_thinking(raw_text)

        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)

        return ModelResponse(
            text=clean_text,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=timer.elapsed_ms,
            finish_reason=output.outputs[0].finish_reason or "unknown",
            raw_response={"model": self.model_id},
            thinking=thinking,
        )

    def unload(self):
        """Free GPU memory."""
        import gc

        import torch

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
