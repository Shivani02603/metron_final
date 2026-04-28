"""
FormAdapter — for form-encoded APIs (application/x-www-form-urlencoded).
If request_template is set, switches to JSON POST (template overrides form encoding).
"""

from __future__ import annotations
import json
import time
import uuid as _uuid_mod
from typing import Any, Dict, List, Optional

import aiohttp

from .chatbot import AdapterResponse


class FormAdapter:
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
        # Use JSON content-type when template is set, form-encoded otherwise
        content_type = "application/json" if request_template else "application/x-www-form-urlencoded"
        self.headers: Dict[str, str] = {"Content-Type": content_type}
        if auth_type == "bearer" and auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    def _build_payload(self, message: str, conversation_id: str):
        if self.request_template:
            escaped = json.dumps(message)[1:-1]
            body_str = (
                self.request_template
                .replace("{{query}}", escaped)
                .replace("{{uuid}}", str(_uuid_mod.uuid4()))
                .replace("{{conversation_id}}", conversation_id or str(_uuid_mod.uuid4()))
            )
            return json.loads(body_str), True  # (payload_dict, is_json)
        return {self.request_field: message}, False

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
        payload, is_json = self._build_payload(message, conversation_id)
        start = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                kwargs = dict(
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                )
                if is_json:
                    kwargs["json"] = payload
                else:
                    kwargs["data"] = payload
                async with session.post(self.endpoint_url, **kwargs) as resp:
                    latency = (time.monotonic() - start) * 1000
                    if resp.status != 200:
                        return AdapterResponse("", latency, error=f"HTTP {resp.status}")
                    try:
                        body: Any = await resp.json(content_type=None)
                    except Exception:
                        body = {"response": await resp.text()}
                    text = self._trim_response(self._extract(body))
                    if text.startswith(("[Field '", "[Empty")):
                        return AdapterResponse("", latency, error=text)
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
                return f"[Field '{p}' not found]"
        return str(result) if result else "[Empty response]"
