"""
Unified LLM client.
Combines:
  - Existing METRON: RateLimiter (token bucket, 20% headroom), 4-provider support,
    token optimization for Azure.
  - New backend: auto-fallback chain on 429/quota exhaustion, JSON extraction
    with retry, proactive throttling.
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import time
from typing import Any, Optional

import litellm

from .config import (
    LLM_PROVIDERS, FALLBACK_CHAIN,
    get_model, resolve_api_key, should_optimize_tokens, get_token_budget,
)

litellm.set_verbose = False


# ── Rate Limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter with 20% headroom.

    Each caller reserves a time slot under the lock (fast), then sleeps
    outside the lock — so multiple callers can sleep concurrently instead
    of queueing behind each other. This makes asyncio.gather() parallelism
    actually useful for rate-limited LLM calls.
    """

    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self.interval = 60.0 / (rpm * 0.8)   # 20% safety headroom
        self.next_allowed = 0.0               # next available call slot
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now >= self.next_allowed:
                # Slot available right now — take it and go
                self.next_allowed = now + self.interval
                return
            # Reserve the next slot; advance the queue pointer
            my_slot = self.next_allowed
            self.next_allowed += self.interval
        # Sleep OUTSIDE the lock — other callers can reserve their slots while we wait
        sleep_time = my_slot - time.monotonic()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


# ── Unified LLM Client ─────────────────────────────────────────────────────

class LLMClient:
    """
    Single entry point for all LLM calls in the pipeline.

    Usage:
        client = LLMClient(provider_name="Groq", api_key="...")
        text = await client.complete(prompt, system="...", temperature=0.7)
        data = await client.complete_json(prompt, system="...", retries=3)
    """

    def __init__(self, provider_name: str = "Groq", api_key: str = ""):
        self.provider_name = provider_name
        self.api_key = resolve_api_key(provider_name, api_key)
        provider_info = LLM_PROVIDERS.get(provider_name, LLM_PROVIDERS["Groq"])
        self.rate_limiter = RateLimiter(provider_info.get("rpm", 30))
        self.optimize_tokens = should_optimize_tokens(provider_name)
        self._exhausted: set[str] = set()   # models exhausted today

    # ── Public API ─────────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 0,
        task: str = "balanced",   # fast | judge | balanced
    ) -> str:
        """Complete a prompt. Returns raw text. Auto-fallback on rate limit."""
        if max_tokens == 0:
            max_tokens = get_token_budget(self.provider_name,
                                          "large" if len(prompt) > 2000 else "normal")

        primary_model = get_model(self.provider_name, task)
        candidates = [primary_model] + [
            m for m in FALLBACK_CHAIN if m != primary_model
        ]

        last_error: Exception = RuntimeError("No models available")
        for model in candidates:
            if model in self._exhausted:
                continue
            for attempt in range(3):
                try:
                    await self.rate_limiter.wait()
                    result = await self._call(model, prompt, system, temperature, max_tokens)
                    return result
                except litellm.exceptions.RateLimitError as e:
                    last_error = e
                    msg = str(e).lower()
                    if "quota" in msg or "resource_exhausted" in msg or "generaterequeststsperday" in msg.replace(" ", ""):
                        self._exhausted.add(model)
                        break   # skip retries, try next model
                    wait = self._parse_retry_after(str(e))
                    await asyncio.sleep(min(wait, 60) + 2)
                except (litellm.exceptions.APIError,
                        litellm.exceptions.ServiceUnavailableError,
                        litellm.exceptions.NotFoundError) as e:
                    last_error = e
                    break   # non-retriable, try next model
                except litellm.exceptions.BadRequestError as e:
                    msg = str(e).lower()
                    if any(k in msg for k in ("429", "quota", "resource_exhausted")):
                        self._exhausted.add(model)
                        break
                    raise   # real bad request — propagate
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        break

        raise RuntimeError(f"All LLM models exhausted. Last error: {last_error}")

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 0,
        task: str = "balanced",
        retries: int = 3,
    ) -> Any:
        """Complete and parse JSON. Retries with clarifying instructions on parse failure."""
        for attempt in range(retries):
            suffix = "" if attempt == 0 else "\n\nReturn ONLY valid JSON. No markdown, no explanation."
            raw = await self.complete(
                prompt + suffix, system=system,
                temperature=temperature, max_tokens=max_tokens, task=task,
            )
            parsed = self._extract_json(raw)
            if parsed is not None:
                return parsed
        raise ValueError(f"Could not extract valid JSON after {retries} attempts. Last response: {raw[:200]}")

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _call(
        self, model: str, prompt: str, system: str,
        temperature: float, max_tokens: int,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Inject API key / endpoint based on provider
        prefix = model.split("/")[0] if "/" in model else ""
        if prefix == "nvidia_nim":
            kwargs["api_key"] = self.api_key or os.environ.get("NVIDIA_NIM_API_KEY", "")
            kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
        elif prefix == "azure":
            kwargs["api_key"] = self.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
            # Strip everything after the hostname — litellm needs only the base URL
            raw_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if raw_endpoint:
                from urllib.parse import urlparse
                parsed = urlparse(raw_endpoint)
                kwargs["api_base"] = f"{parsed.scheme}://{parsed.netloc}/"
            kwargs["api_version"] = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
        elif prefix == "groq":
            kwargs["api_key"] = self.api_key if "groq" in self.provider_name.lower() else os.environ.get("GROQ_API_KEY", "")
        elif prefix == "gemini":
            kwargs["api_key"] = self.api_key if "gemini" in self.provider_name.lower() else os.environ.get("GEMINI_API_KEY", "")

        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return None
        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2. Markdown code block
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        # 3. Greedy brace/bracket extraction
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            if start != -1:
                end = text.rfind(end_char)
                if end > start:
                    try:
                        return json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        pass
        return None

    @staticmethod
    def _parse_retry_after(error_msg: str) -> float:
        m = re.search(r"retry.?after[:\s]+(\d+(?:\.\d+)?)", error_msg, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*second", error_msg, re.I)
        if m:
            return float(m.group(1))
        return 10.0   # default backoff
