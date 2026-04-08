"""
FormAdapter — for form-encoded APIs (application/x-www-form-urlencoded).
Sourced from new metron-backend/app/stage4_execution/adapters/form_adapter.py.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .chatbot import AdapterResponse


class FormAdapter:
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
        self.headers: Dict[str, str] = {"Content-Type": "application/x-www-form-urlencoded"}
        if auth_type == "bearer" and auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    async def send(self, message: str, history: Optional[List] = None) -> AdapterResponse:
        data = {self.request_field: message}
        start = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint_url,
                    data=data,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    latency = (time.monotonic() - start) * 1000
                    if resp.status != 200:
                        return AdapterResponse("", latency, error=f"HTTP {resp.status}")
                    try:
                        body: Any = await resp.json(content_type=None)
                    except Exception:
                        body = {"response": await resp.text()}
                    text = self._extract(body)
                    return AdapterResponse(text, latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse("", latency, error=str(e))

    def _extract(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        parts = self.response_field.split(".")
        result: Any = data
        for p in parts:
            if isinstance(result, dict) and p in result:
                result = result[p]
            else:
                return "[Field not found]"
        return str(result) if result else "[Empty]"
