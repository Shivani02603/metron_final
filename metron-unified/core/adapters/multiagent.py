"""
MultiAgentAdapter — for multi-agent systems.
Captures agent_trace (which agents ran, tools called, sub-latencies).
Sourced from new metron-backend/app/stage4_execution/adapters/multiagent_adapter.py.
"""

from __future__ import annotations
import time
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
        endpoint_url:   str,
        request_field:  str = "message",
        response_field: str = "response",
        auth_type:      str = "none",
        auth_token:     str = "",
        timeout:        int = 60,   # longer timeout for multi-agent
    ):
        self.endpoint_url   = endpoint_url
        self.request_field  = request_field
        self.response_field = response_field
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
                    text  = self._extract_text(data)
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
            else:
                break
        if result is not None and result != data:
            return str(result)
        # Fallback: try common final-answer fields
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
