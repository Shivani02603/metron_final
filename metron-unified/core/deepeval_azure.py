"""
Custom DeepEval model wrapper for Azure OpenAI.

DeepEval's built-in metric constructors accept model= as either:
  - a plain string like "gpt-4o" (routed to OpenAI)
  - a DeepEvalBaseLLM subclass instance (custom routing)

Passing "azure/gpt-4o" fails DeepEval's internal model-name validation.
This class wraps the Azure OpenAI client so all DeepEval metrics (GEval,
HallucinationMetric, AnswerRelevancyMetric, BiasMetric) route through Azure.

Usage:
    from core.deepeval_azure import make_deepeval_azure_model
    model = make_deepeval_azure_model()
    metric = HallucinationMetric(threshold=0.5, model=model)
"""

from __future__ import annotations
import os
from typing import Optional
from urllib.parse import urlparse


def _parse_deployment(endpoint: str) -> str:
    """Extract deployment name from Azure OpenAI endpoint URL."""
    parsed = urlparse(endpoint)
    parts  = [p for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("deployments")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "gpt-4o"


def _parse_base_url(endpoint: str) -> str:
    """Extract base URL (scheme + netloc) from Azure OpenAI endpoint URL."""
    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}/"


class _DeepEvalAzureOpenAI:
    """
    Minimal DeepEvalBaseLLM subclass that routes through Azure OpenAI.
    Defined as an inner class so the import of deepeval is deferred —
    avoids import-time side effects when deepeval is not installed.
    """
    pass


def make_deepeval_azure_model():
    """
    Build and return a DeepEvalBaseLLM-compatible Azure OpenAI model instance.
    Returns None if deepeval or openai is not installed.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM
        from openai import AzureOpenAI
    except ImportError:
        return None

    endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key    = os.environ.get("AZURE_OPENAI_API_KEY", "")
    api_version= os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
    deployment = _parse_deployment(endpoint)
    base_url   = _parse_base_url(endpoint)

    class AzureGPT4oModel(DeepEvalBaseLLM):
        """Azure OpenAI GPT-4o model for DeepEval evaluation metrics."""

        def __init__(self):
            # Set attributes before super().__init__ because that calls load_model()
            self._deployment  = deployment
            self._base_url    = base_url
            self._api_key     = api_key
            self._api_version = api_version
            super().__init__(model_name=f"azure-{deployment}")

        def load_model(self):
            return AzureOpenAI(
                azure_endpoint=self._base_url,
                api_key=self._api_key,
                api_version=self._api_version,
            )

        def generate(self, prompt: str) -> str:
            import time
            last_exc = None
            for attempt in range(3):
                try:
                    response = self.model.chat.completions.create(
                        model=self._deployment,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=2048,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
                        wait = 10 * (attempt + 1)
                        print(f"[DeepEval/Azure] Rate limited — waiting {wait}s (attempt {attempt+1}/3)")
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError(f"DeepEval Azure: max retries exceeded — {last_exc}")

        async def a_generate(self, prompt: str) -> str:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.generate, prompt)

        def get_model_name(self) -> str:
            return f"Azure/{self._deployment}"

    return AzureGPT4oModel()
