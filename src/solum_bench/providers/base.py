"""Base protocol and data types for model providers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ModelResponse:
    """Response from a model provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str
    raw_response: dict = field(default_factory=dict)
    thinking: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class GenerateRequest:
    """A single generation request."""

    system: str
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: int | None = None


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol for model providers."""

    name: str
    provider_type: str

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse: ...


def extract_thinking(text: str) -> tuple[str | None, str]:
    """Extract <think>...</think> blocks from model output.

    Returns (thinking_content, clean_response).
    """
    pattern = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
    match = pattern.search(text)
    if match:
        thinking = match.group(1).strip()
        clean = pattern.sub("", text).strip()
        return thinking, clean
    return None, text


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
