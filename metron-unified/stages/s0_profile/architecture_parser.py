"""
Architecture document / diagram parser.

Accepts either:
  - Plain text / PDF-extracted text describing the architecture
  - Base64-encoded image (architecture diagram) — parsed via vision LLM

Returns a dict of structured architecture fields that map directly to
RunConfig's architecture profile section, plus enriched
additional_architecture_notes for the RCA keyword mapper.
"""

from __future__ import annotations
import base64
import re
from typing import Any, Dict

from core.llm_client import LLMClient

# ── Extraction prompt (text document) ─────────────────────────────────────

_TEXT_SYSTEM = """You are an expert infrastructure architect.
Extract deployment and infrastructure details from a document.
Return ONLY valid JSON. No markdown, no explanation."""

_TEXT_PROMPT = """
Analyse this architecture document and extract infrastructure details.

DOCUMENT:
{document}

Return JSON with exactly these fields (use empty string "" if not mentioned):
{{
  "deployment_type":   "serverless" | "server" | "container" | "unknown",
  "vector_db":         "pinecone" | "weaviate" | "qdrant" | "faiss" | "chroma" | "other" | "",
  "session_db":        "redis" | "postgresql" | "mongodb" | "dynamodb" | "sqlite" | "other" | "",
  "cache_layer":       "redis" | "memcached" | "cdn" | "in_memory" | "other" | "",
  "message_queue":     "kafka" | "rabbitmq" | "sqs" | "pubsub" | "kinesis" | "other" | "",
  "api_gateway":       "aws_apigw" | "azure_apim" | "kong" | "nginx" | "other" | "",
  "auth_mechanism":    "oauth" | "api_key" | "jwt" | "saml" | "other" | "",
  "monitoring_tool":   "datadog" | "cloudwatch" | "prometheus" | "grafana" | "newrelic" | "other" | "",
  "is_multi_region":   true | false,
  "has_rate_limiting": true | false,
  "has_retry_logic":   true | false,
  "has_circuit_breaker": true | false,
  "has_caching":       true | false,
  "has_dlq":           true | false,
  "summary":           "<2-3 sentence plain-English summary of the architecture>"
}}

Rules:
- deployment_type: "serverless" if Lambda/Azure Functions/Cloud Run mentioned;
  "container" if Kubernetes/Docker/ECS; "server" if EC2/VM; else "unknown"
- Set boolean to true only if explicitly mentioned or clearly implied
- summary: describe key components in plain English for the RCA notes field
"""

# ── Vision prompt (diagram image) ─────────────────────────────────────────

_VISION_SYSTEM = """You are an expert infrastructure architect.
Analyse architecture diagram images and extract deployment details.
Return ONLY valid JSON. No markdown, no explanation."""

_VISION_PROMPT = """
This is an architecture diagram. Extract all infrastructure components and
deployment details visible in the diagram.

Return JSON with exactly these fields (use empty string "" if not visible):
{{
  "deployment_type":   "serverless" | "server" | "container" | "unknown",
  "vector_db":         "pinecone" | "weaviate" | "qdrant" | "faiss" | "chroma" | "other" | "",
  "session_db":        "redis" | "postgresql" | "mongodb" | "dynamodb" | "sqlite" | "other" | "",
  "cache_layer":       "redis" | "memcached" | "cdn" | "in_memory" | "other" | "",
  "message_queue":     "kafka" | "rabbitmq" | "sqs" | "pubsub" | "kinesis" | "other" | "",
  "api_gateway":       "aws_apigw" | "azure_apim" | "kong" | "nginx" | "other" | "",
  "auth_mechanism":    "oauth" | "api_key" | "jwt" | "saml" | "other" | "",
  "monitoring_tool":   "datadog" | "cloudwatch" | "prometheus" | "grafana" | "newrelic" | "other" | "",
  "is_multi_region":   true | false,
  "has_rate_limiting": true | false,
  "has_retry_logic":   true | false,
  "has_circuit_breaker": true | false,
  "has_caching":       true | false,
  "has_dlq":           true | false,
  "components_found":  ["<list of all components/services visible in the diagram>"],
  "summary":           "<2-3 sentence plain-English summary of the architecture>"
}}
"""

# ── Vision-capable providers ───────────────────────────────────────────────

_VISION_PROVIDERS = {
    "Azure OpenAI",
    "Gemini",
    "OpenAI",
}


async def parse_architecture_text(
    text: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Extract structured architecture fields from a text document.
    Returns dict of architecture fields + 'summary' key.
    """
    prompt = _TEXT_PROMPT.format(document=text[:6000])
    try:
        result = await llm_client.complete_json(
            prompt, system=_TEXT_SYSTEM, temperature=0.1, task="fast"
        )
        return _normalise(result)
    except Exception as e:
        return {"error": str(e), "summary": text[:500]}


async def parse_architecture_image(
    image_b64: str,
    mime_type: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Extract structured architecture fields from a base64-encoded image.
    Uses vision LLM if provider supports it; falls back to error message otherwise.

    image_b64 : base64-encoded image bytes
    mime_type : e.g. "image/png", "image/jpeg", "image/webp"
    """
    if llm_client.provider_name not in _VISION_PROVIDERS:
        return {
            "error": f"Vision not supported for provider '{llm_client.provider_name}'. "
                     f"Switch to Azure OpenAI, Gemini, or OpenAI for diagram parsing.",
            "summary": "",
        }

    # Build vision message
    import litellm
    from core.config import get_model, resolve_api_key
    import os

    model = get_model(llm_client.provider_name, "balanced")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
            ],
        }
    ]

    kwargs: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.1,
        "max_tokens":  1500,
    }

    prefix = model.split("/")[0] if "/" in model else ""
    if prefix == "azure":
        kwargs["api_key"]     = llm_client.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        raw_ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if raw_ep:
            from urllib.parse import urlparse
            p = urlparse(raw_ep)
            kwargs["api_base"] = f"{p.scheme}://{p.netloc}/"
        kwargs["api_version"] = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
    elif prefix == "gemini":
        kwargs["api_key"] = llm_client.api_key or os.environ.get("GEMINI_API_KEY", "")

    try:
        await llm_client.rate_limiter.wait()
        response = await litellm.acompletion(**kwargs)
        raw = response.choices[0].message.content or ""
        result = llm_client._extract_json(raw)
        if result:
            return _normalise(result)
        return {"error": "Could not parse JSON from vision response", "raw_description": raw[:1000]}
    except Exception as e:
        return {"error": f"Vision parse failed: {str(e)[:200]}"}


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected keys exist with correct types."""
    str_fields = [
        "deployment_type", "vector_db", "session_db", "cache_layer",
        "message_queue", "api_gateway", "auth_mechanism", "monitoring_tool", "summary",
    ]
    bool_fields = [
        "is_multi_region", "has_rate_limiting", "has_retry_logic",
        "has_circuit_breaker", "has_caching", "has_dlq",
    ]
    out: Dict[str, Any] = {}
    for f in str_fields:
        out[f] = str(data.get(f, "")).strip()
    for f in bool_fields:
        val = data.get(f, False)
        out[f] = bool(val) if isinstance(val, bool) else str(val).lower() in ("true", "yes", "1")
    out["components_found"] = data.get("components_found", [])
    return out
