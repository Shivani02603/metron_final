"""
ChatbotAdapter — generic JSON REST chatbot.
Supports dot-notation response field extraction (e.g. "output.text.value").
Supports full JSON request templates with {{query}}, {{uuid}}, {{conversation_id}} placeholders.
"""

from __future__ import annotations
import json
import time
import uuid as _uuid_mod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp


@dataclass
class AdapterResponse:
    text:              str
    latency_ms:        float
    retrieved_context: Optional[List[str]] = None
    agent_trace:       Optional[List[Dict]] = None
    error:             Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.text.startswith("[Error")


class ChatbotAdapter:
    """
    Sends {request_field: message} → reads {response_field: answer}.
    If request_template is set, renders the full JSON body instead.
    Supports nested response fields with dot notation.
    """

    def __init__(
        self,
        endpoint_url:         str,
        request_field:        str = "message",
        response_field:       str = "response",
        auth_type:            str = "none",
        auth_token:           str = "",
        timeout:              int = 30,
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
            # json.dumps escapes newlines, tabs, quotes etc; strip outer quotes to get bare escaped string
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
                    text = self._trim_response(self._extract(data, self.response_field))
                    # Extraction errors go into the error field so evaluators
                    # never receive internal sentinel strings as real responses.
                    if text.startswith(("[Field ", "[Index ", "[Empty")):
                        raw_hint = json.dumps(data, ensure_ascii=False)[:500]
                        return AdapterResponse("", latency, error=f"{text}\nActual response: {raw_hint}")
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
        # Support fallback paths separated by | (e.g. for A2A protocol where
        # completed responses have artifacts but in-progress use status.message)
        paths = [p.strip() for p in field_path.split("|") if p.strip()]
        last_error = f"[Field '' not found]"
        for path in paths:
            result = ChatbotAdapter._extract_single(data, path)
            if not result.startswith(("[Field ", "[Index ", "[Empty")):
                return result
            last_error = result
        return last_error

    @staticmethod
    def _extract_single(data: dict, field_path: str) -> str:
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
