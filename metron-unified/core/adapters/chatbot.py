"""
ChatbotAdapter — generic JSON REST chatbot.
Sourced directly from existing METRON ChatbotAdapter (app_v3.py).
Supports dot-notation response field extraction (e.g. "output.text.value").
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp


@dataclass
class AdapterResponse:
    text:             str
    latency_ms:       float
    retrieved_context: Optional[List[str]] = None
    agent_trace:      Optional[List[Dict]] = None
    error:            Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.text.startswith("[Error")


class ChatbotAdapter:
    """
    Sends {request_field: message} → reads {response_field: answer}.
    Supports nested response fields with dot notation.
    """

    def __init__(
        self,
        endpoint_url:   str,
        request_field:  str = "message",
        response_field: str = "response",
        auth_type:      str = "none",
        auth_token:     str = "",
        timeout:        int = 30,
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
                    text = self._extract(data, self.response_field)
                    return AdapterResponse(text, latency)
        except aiohttp.ClientConnectorError as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse("", latency, error=f"Connection refused: {e}")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse("", latency, error=str(e))

    async def test_connection(self) -> Tuple[bool, str]:
        resp = await self.send("Hello, this is a connectivity test.")
        if resp.ok:
            return True, f"Connected. Latency: {resp.latency_ms:.0f}ms"
        return False, resp.error or "Unknown error"

    @staticmethod
    def _extract(data: dict, field_path: str) -> str:
        parts = field_path.split(".")
        result: object = data
        for part in parts:
            if isinstance(result, dict) and part in result:
                result = result[part]
            elif isinstance(result, list) and part.isdigit():
                idx = int(part)
                result = result[idx] if 0 <= idx < len(result) else "[Index OOB]"
            else:
                return f"[Field '{part}' not found]"
        return str(result) if result is not None else "[Empty response]"
