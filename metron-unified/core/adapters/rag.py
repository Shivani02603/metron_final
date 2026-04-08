"""
RAGAdapter — for RAG systems.
Sends message, auto-discovers and extracts retrieved_context from response.
Sourced from new metron-backend/app/stage4_execution/adapters/rag_adapter.py.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .chatbot import AdapterResponse

# Common field names RAG systems use for retrieved context
_CONTEXT_FIELD_CANDIDATES = [
    "retrieved_context", "context", "sources", "documents",
    "chunks", "passages", "references", "source_documents",
]


class RAGAdapter:
    def __init__(
        self,
        endpoint_url:   str,
        request_field:  str = "message",
        response_field: str = "response",
        auth_type:      str = "none",
        auth_token:     str = "",
        context_field:  str = "",   # explicit override; auto-detected if empty
        timeout:        int = 30,
    ):
        self.endpoint_url   = endpoint_url
        self.request_field  = request_field
        self.response_field = response_field
        self.context_field  = context_field
        self.timeout        = timeout
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_type == "bearer" and auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    async def send(self, message: str, history: Optional[List] = None) -> AdapterResponse:
        payload = {self.request_field: message}
        start = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    latency = (time.monotonic() - start) * 1000
                    if resp.status != 200:
                        return AdapterResponse("", latency, error=f"HTTP {resp.status}")
                    data = await resp.json(content_type=None)
                    text = self._extract_text(data)
                    context = self._extract_context(data)
                    return AdapterResponse(text, latency, retrieved_context=context)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse("", latency, error=str(e))

    def _extract_text(self, data: Dict[str, Any]) -> str:
        parts = self.response_field.split(".")
        result: Any = data
        for p in parts:
            if isinstance(result, dict) and p in result:
                result = result[p]
            else:
                return "[Response field not found]"
        return str(result) if result else "[Empty]"

    def _extract_context(self, data: Dict[str, Any]) -> Optional[List[str]]:
        # Explicit field override
        if self.context_field and self.context_field in data:
            raw = data[self.context_field]
            return self._normalise_context(raw)
        # Auto-detect
        for candidate in _CONTEXT_FIELD_CANDIDATES:
            if candidate in data:
                return self._normalise_context(data[candidate])
        return None

    @staticmethod
    def _normalise_context(raw: Any) -> List[str]:
        if isinstance(raw, list):
            result = []
            for item in raw:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    # Try common text fields
                    for key in ("text", "content", "page_content", "chunk"):
                        if key in item:
                            result.append(str(item[key]))
                            break
            return result
        if isinstance(raw, str):
            return [raw]
        return []
