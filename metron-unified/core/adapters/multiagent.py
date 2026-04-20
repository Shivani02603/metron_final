"""
MultiAgentAdapter — for multi-agent systems.
Captures agent_trace (which agents ran, tools called, sub-latencies).
Supports request templates and response trim marker.
"""

from __future__ import annotations
import json
import time
import uuid as _uuid_mod
from typing import Any, Dict, List, Optional

import aiohttp

from .chatbot import AdapterResponse

_TRACE_FIELD_CANDIDATES = [
    "agent_trace", "trace", "steps", "intermediate_steps",
    "tool_calls", "agents", "execution_trace",
]
_FINAL_ANSWER_CANDIDATES = [
    "final_answer", "answer", "response", "output", "result",
]


class MultiAgentAdapter:
    def __init__(
        self,
        endpoint_url:         str,
        request_field:        str = "message",
        response_field:       str = "response",
        auth_type:            str = "none",
        auth_token:           str = "",
        timeout:              int = 60,
        request_template:     Optional[str] = None,
        response_trim_marker: Optional[str] = None,
    ):
        self.endpoint_url         = endpoint_url
        self.request_field        = request_field
        self.response_field       = response_field
        self.timeout              = timeout
        self.request_template     = request_template
        self.response_trim_marker = response_trim_marker
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if auth_type == "bearer" and auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    def _build_payload(self, message: str, conversation_id: str) -> dict:
        if self.request_template:
            escaped = json.dumps(message)[1:-1]
            body_str = (
                self.request_template
                .replace("{{query}}", escaped)
                .replace("{{uuid}}", str(_uuid_mod.uuid4()))
                .replace("{{conversation_id}}", conversation_id or str(_uuid_mod.uuid4()))
            )
            return json.loads(body_str)
        return {self.request_field: message}

    def _trim_response(self, text: str) -> str:
        if self.response_trim_marker and self.response_trim_marker in text:
            return text[:text.index(self.response_trim_marker)].rstrip()
        return text

    async def send(
        self,
        message: str,
        history: Optional[List] = None,
        conversation_id: str = "",
    ) -> AdapterResponse:
        payload = self._build_payload(message, conversation_id)
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
                    text  = self._trim_response(self._extract_text(data))
                    trace = self._extract_trace(data)
                    return AdapterResponse(text, latency, agent_trace=trace)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse("", latency, error=str(e))

    def _extract_text(self, data: Dict[str, Any]) -> str:
        parts = self.response_field.split(".")
        result: Any = data
        for p in parts:
            if isinstance(result, dict) and p in result:
                result = result[p]
            elif isinstance(result, list) and p.isdigit():
                idx = int(p)
                result = result[idx] if 0 <= idx < len(result) else None
                if result is None:
                    break
            else:
                break
        if result is not None and result != data:
            return str(result)
        for key in _FINAL_ANSWER_CANDIDATES:
            if key in data:
                return str(data[key])
        return "[No final answer field found]"

    @staticmethod
    def _extract_trace(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        for candidate in _TRACE_FIELD_CANDIDATES:
            if candidate in data:
                raw = data[candidate]
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, dict):
                    return [raw]
        return None
