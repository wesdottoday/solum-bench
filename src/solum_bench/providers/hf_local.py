"""HuggingFace transformers local provider."""

from __future__ import annotations

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class HFLocalProvider:
    """Provider for local models via HuggingFace transformers."""

    provider_type = "hf_local"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        **kwargs,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install 'solum-bench[hf]'")

        self.name = name or model_id
        self.model_id = model_id

        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        import torch

        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.extend({"role": m["role"], "content": m["content"]} for m in messages)

        input_text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        if seed is not None:
            torch.manual_seed(seed)

        with Timer() as timer:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        thinking, clean_text = extract_thinking(raw_text)

        return ModelResponse(
            text=clean_text,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
            latency_ms=timer.elapsed_ms,
            finish_reason="stop",
            raw_response={"model": self.model_id},
            thinking=thinking,
        )
