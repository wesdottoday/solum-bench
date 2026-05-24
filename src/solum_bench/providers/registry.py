"""Provider registry and factory for instantiating model providers from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from solum_bench.providers.base import ModelProvider


def create_provider(config: dict[str, Any]) -> ModelProvider:
    """Create a model provider from a config dict."""
    provider_type = config["provider"]
    model_id = config["model_id"]
    name = config.get("name", model_id)

    # Filter out keys consumed by the registry
    extra = {k: v for k, v in config.items() if k not in ("provider", "model_id", "name")}

    if provider_type == "anthropic":
        from solum_bench.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model_id=model_id, name=name, **extra)

    elif provider_type == "openai":
        from solum_bench.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            model_id=model_id,
            name=name,
            api_base=extra.pop("api_base", None),
            api_key=extra.pop("api_key", None),
            **extra,
        )

    elif provider_type == "google":
        from solum_bench.providers.google import GoogleProvider

        return GoogleProvider(model_id=model_id, name=name, **extra)

    elif provider_type == "vllm_local":
        from solum_bench.providers.vllm_local import VLLMLocalProvider

        return VLLMLocalProvider(model_id=model_id, name=name, **extra)

    elif provider_type == "hf_local":
        from solum_bench.providers.hf_local import HFLocalProvider

        return HFLocalProvider(model_id=model_id, name=name, **extra)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def load_providers_from_config(config_path: Path) -> list[ModelProvider]:
    """Load all model providers from a YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    providers = []
    for model_config in config.get("models", []):
        providers.append(create_provider(model_config))
    return providers
