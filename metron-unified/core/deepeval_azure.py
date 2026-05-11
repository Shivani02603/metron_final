"""
DeepEval model wrapper backed by LiteLLM.

Works with all providers in core/config.py (Gemini, Groq, Azure, NVIDIA NIM).
LiteLLM normalises every provider's responses to OpenAI format, which is what
DeepEval's internal parsers expect — so all metrics work without changes.

Usage:
    from core.deepeval_azure import make_deepeval_model
    model = make_deepeval_model(provider_name, api_key, azure_endpoint)
    metric = HallucinationMetric(threshold=0.5, model=model)

Backward compat:
    make_deepeval_azure_model() still works — it delegates to make_deepeval_model()
    using Azure credentials from environment variables.
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional
from urllib.parse import urlparse


def make_deepeval_model(
    provider_name: str = "Azure OpenAI",
    api_key: str = "",
    azure_endpoint: str = "",
) -> Optional[object]:
    """
    Build a DeepEvalBaseLLM-compatible model for any configured provider via LiteLLM.
    Returns None if deepeval/litellm is not installed or credentials are missing.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM  # noqa: F401
    except ImportError:
        return None

    try:
        import litellm  # noqa: F401
    except ImportError:
        return None

    from core.config import get_model, resolve_api_key

    resolved_key = api_key or resolve_api_key(provider_name)
    model_str = get_model(provider_name, task="judge")
    prefix = model_str.split("/")[0] if "/" in model_str else ""

    extra_kwargs: dict = {}

    if prefix == "azure":
        resolved_key = resolved_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not resolved_key:
            return None
        raw = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if raw:
            parsed = urlparse(raw)
            extra_kwargs["api_base"] = f"{parsed.scheme}://{parsed.netloc}/"
        extra_kwargs["api_version"] = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")

    elif prefix == "nvidia_nim":
        resolved_key = resolved_key or os.environ.get("NVIDIA_NIM_API_KEY", "")
        if not resolved_key:
            return None
        extra_kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"

    elif prefix == "groq":
        resolved_key = resolved_key or os.environ.get("GROQ_API_KEY", "")
        if not resolved_key:
            return None

    elif prefix == "gemini":
        resolved_key = resolved_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            return None

    else:
        if not resolved_key:
            return None

    # Capture in local vars for the closure — avoids late-binding issues
    _model = model_str
    _key = resolved_key
    _extra = dict(extra_kwargs)

    from deepeval.models import DeepEvalBaseLLM

    class _LiteLLMDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(model_name=_model)

        def load_model(self):
            # LiteLLM is a stateless module — no client object is needed.
            return None

        def generate(self, prompt: str) -> str:
            import time
            import litellm
            kwargs: dict = {
                "model": _model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2048,
                "api_key": _key,
                **_extra,
            }
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(3):
                try:
                    response = litellm.completion(**kwargs)
                    return response.choices[0].message.content or ""
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
                        time.sleep(10 * (attempt + 1))
                    else:
                        raise
            raise RuntimeError(f"DeepEval LiteLLM: max retries exceeded — {last_exc}")

        async def a_generate(self, prompt: str) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.generate, prompt)

        def get_model_name(self) -> str:
            return _model

    return _LiteLLMDeepEvalModel()


def make_deepeval_azure_model():
    """Backward-compatible alias — delegates to make_deepeval_model() with Azure env vars."""
    return make_deepeval_model(
        provider_name="Azure OpenAI",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    )
