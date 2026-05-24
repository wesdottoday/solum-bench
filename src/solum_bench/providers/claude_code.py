"""Claude Code CLI provider — uses 'claude -p' for inference without API keys."""

from __future__ import annotations

import json
import subprocess

from solum_bench.providers.base import ModelResponse, Timer, extract_thinking


class ClaudeCodeProvider:
    """Provider that shells out to the Claude Code CLI ('claude -p').

    Uses the same authentication as the user's Claude Code session,
    so no API key is needed. Works with enterprise billing.
    """

    provider_type = "claude_code"

    def __init__(self, name: str | None = None, model: str | None = None, **kwargs):
        self.name = name or "claude-code"
        self._model_flag = model  # e.g. "sonnet" or "opus" — passed as --model

        # Verify claude CLI is available
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI returned exit code {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Make sure 'claude' is on your PATH.\n"
                "Install: https://docs.anthropic.com/en/docs/claude-code"
            )

    def generate(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        # Build the prompt — combine system + user messages
        prompt_parts = []
        if system:
            prompt_parts.append(f"[System instructions: {system}]\n")
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                prompt_parts.append(f"[Previous assistant response: {msg['content']}]")
        full_prompt = "\n\n".join(prompt_parts)

        cmd = ["claude", "-p", "--output-format", "json", "--tools", ""]
        if self._model_flag:
            cmd.extend(["--model", self._model_flag])
        cmd.extend(["--max-turns", "1"])

        with Timer() as timer:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )

        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")

        # Parse JSON output from claude -p --output-format json
        try:
            data = json.loads(result.stdout)
            raw_text = data.get("result", result.stdout)
            duration_ms = data.get("duration_api_ms", timer.elapsed_ms)
            cost_usd = data.get("total_cost_usd", 0)
            stop_reason = data.get("stop_reason", "stop")

            # Extract token counts from usage block
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Get actual model name from modelUsage
            model_usage = data.get("modelUsage", {})
            model_id = next(iter(model_usage.keys()), self.name) if model_usage else self.name

        except (json.JSONDecodeError, KeyError):
            raw_text = result.stdout.strip()
            input_tokens = 0
            output_tokens = 0
            duration_ms = timer.elapsed_ms
            cost_usd = 0
            stop_reason = "stop"
            model_id = self.name

        thinking, clean_text = extract_thinking(raw_text)

        return ModelResponse(
            text=clean_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=duration_ms,
            finish_reason=stop_reason,
            raw_response={"model": model_id, "provider": "claude_code", "cost_usd": cost_usd},
            thinking=thinking,
        )
