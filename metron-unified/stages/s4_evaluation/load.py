"""
Stage 4e: Load test evaluation using Locust.
Runs Locust headless as a subprocess — avoids gevent/asyncio conflicts.

Locust is invoked via: python -m locust --headless ...
Results are read from Locust's CSV output (--csv flag).

If locust is not installed, raises ImportError with a clear message.
"""

from __future__ import annotations
import asyncio
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from core.models import RunConfig

LOAD_TEST_PROMPTS = [
    "Hello, can you help me?",
    "What are your capabilities?",
    "I need assistance with a task.",
    "Can you explain your main features?",
    "How do you handle complex or unusual requests?",
]

# ── Locust file template ───────────────────────────────────────────────────────
# Values are injected via json.dumps for safe string encoding.

_LOCUST_FILE_TEMPLATE = textwrap.dedent("""\
    import json
    import random
    from locust import HttpUser, task, between

    _PROMPTS        = {prompts_json}
    _REQUEST_FIELD  = {request_field_json}
    _RESPONSE_FIELD = {response_field_json}
    _AUTH_TYPE      = {auth_type_json}
    _AUTH_TOKEN     = {auth_token_json}
    _PATH           = {path_json}


    class AIEndpointUser(HttpUser):
        wait_time = between(1.0, 3.0)

        def on_start(self):
            if _AUTH_TYPE == "bearer" and _AUTH_TOKEN:
                self.client.headers.update({{"Authorization": f"Bearer {{_AUTH_TOKEN}}"}})

        @task
        def send_request(self):
            prompt  = random.choice(_PROMPTS)
            payload = {{_REQUEST_FIELD: prompt}}
            with self.client.post(
                _PATH, json=payload, catch_response=True, name="AI Endpoint"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if _RESPONSE_FIELD in data:
                            response.success()
                        else:
                            response.failure(
                                f"Field '{{_RESPONSE_FIELD}}' not found in response"
                            )
                    except Exception as exc:
                        response.failure(f"JSON parse error: {{exc}}")
                else:
                    response.failure(f"HTTP {{response.status_code}}")
""")


def _build_locust_file(config: RunConfig, path: str) -> str:
    """
    Write a parameterised Locust file to `path`.
    Returns the host URL (scheme + netloc) for --host flag.
    """
    parsed   = urlparse(config.endpoint_url)
    host     = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = parsed.path or "/"

    content = _LOCUST_FILE_TEMPLATE.format(
        prompts_json       = json.dumps(LOAD_TEST_PROMPTS),
        request_field_json = json.dumps(config.request_field),
        response_field_json= json.dumps(config.response_field),
        auth_type_json     = json.dumps(config.auth_type),
        auth_token_json    = json.dumps(config.auth_token),
        path_json          = json.dumps(endpoint),
    )
    Path(path).write_text(content, encoding="utf-8")
    return host


def _parse_locust_csv(csv_prefix: str) -> Dict[str, Any]:
    """
    Parse Locust's *_stats.csv output file and extract the Aggregated row.
    Returns a metrics dict.
    """
    stats_file = f"{csv_prefix}_stats.csv"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Locust CSV not found: {stats_file}")

    with open(stats_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name", "").strip().lower() == "aggregated":
                total     = int(row.get("Request Count", 0) or 0)
                failures  = int(row.get("Failure Count", 0) or 0)
                avg_ms    = float(row.get("Average Response Time", 0) or 0)
                p95_ms    = float(row.get("95%", 0) or 0)
                p99_ms    = float(row.get("99%", 0) or 0)
                rps       = float(row.get("Requests/s", 0) or 0)
                error_rate = round(failures / total * 100, 2) if total > 0 else 0.0

                return {
                    "tool_used":           "locust",
                    "total_requests":      total,
                    "successful":          total - failures,
                    "errors":              failures,
                    "error_rate":          error_rate,
                    "avg_latency_ms":      round(avg_ms, 2),
                    "p95_latency_ms":      round(p95_ms, 2),
                    "p99_latency_ms":      round(p99_ms, 2),
                    "requests_per_second": round(rps, 3),
                    "passed": p95_ms <= 5000 and error_rate < 5.0,
                    "assessment":          _assess_load(error_rate / 100, p95_ms),
                }

    raise ValueError("Locust CSV found but 'Aggregated' row missing.")


def _assess_load(error_rate: float, p95_ms: float) -> str:
    if error_rate < 0.01 and p95_ms < 2000:
        return "excellent"
    if error_rate < 0.05 and p95_ms < 5000:
        return "acceptable"
    if error_rate < 0.10:
        return "degraded"
    return "critical"


# ── Preflight check ───────────────────────────────────────────────────────────

async def _preflight_check(config: RunConfig) -> None:
    """
    Send one test request to the endpoint before starting Locust.
    Logs the HTTP status, response body snippet, and detected field names so
    the user can see immediately if the endpoint / request_field / response_field
    settings are wrong — instead of only learning this from a 100% Locust error rate.
    Does NOT raise — Locust still runs even if preflight fails.
    """
    try:
        payload  = json.dumps({config.request_field: LOAD_TEST_PROMPTS[0]}).encode()
        headers  = {"Content-Type": "application/json"}
        if config.auth_type == "bearer" and config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"

        req = urllib.request.Request(
            config.endpoint_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body    = resp.read(2000).decode(errors="replace")
            status  = resp.status
            try:
                data   = json.loads(body)
                fields = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            except Exception:
                data   = None
                fields = "(non-JSON response)"

            if isinstance(data, dict) and config.response_field not in data:
                print(
                    f"[Load/Preflight] WARNING: response_field='{config.response_field}' "
                    f"not found in response. Actual keys: {fields}. "
                    f"Locust will report all requests as failures — "
                    f"update response_field to one of: {fields}"
                )
            else:
                print(
                    f"[Load/Preflight] OK — endpoint reachable, HTTP {status}, "
                    f"response_field='{config.response_field}' present. "
                    f"Response snippet: {body[:200]}"
                )
    except urllib.error.HTTPError as e:
        body = e.read(500).decode(errors="replace")
        print(
            f"[Load/Preflight] HTTP {e.code} from endpoint — Locust will likely get 100% errors. "
            f"Response body: {body[:300]}"
        )
    except Exception as e:
        print(f"[Load/Preflight] Could not reach endpoint: {e} — Locust may fail entirely.")


# ── Main evaluator ─────────────────────────────────────────────────────────────

async def evaluate_load(config: RunConfig) -> Dict[str, Any]:
    """
    Run Locust headless as a subprocess against config.endpoint_url.
    Returns aggregate load metrics dict.
    Raises if locust is not installed or the subprocess fails.
    """
    # Verify locust is installed before doing any file I/O
    import importlib.util
    if importlib.util.find_spec("locust") is None:
        raise ImportError(
            "Locust is not installed. Run: pip install locust>=2.20.0"
        )

    num_users    = config.load_concurrent_users
    duration_s   = config.load_duration_seconds

    # ── Preflight: one test request to surface endpoint errors early ──────────
    await _preflight_check(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        locust_file  = os.path.join(tmp_dir, "locustfile.py")
        csv_prefix   = os.path.join(tmp_dir, "locust_results")

        host = _build_locust_file(config, locust_file)

        # Gradual spawn: ramp up 1/5th of users per second (spreads load, avoids thundering herd)
        spawn_rate = max(1, num_users // 5)

        cmd = [
            sys.executable, "-m", "locust",
            "--headless",
            "--host",        host,
            "--users",       str(num_users),
            "--spawn-rate",  str(spawn_rate),
            "--run-time",    f"{duration_s}s",
            "--csv",         csv_prefix,
            "--exit-code-on-error", "0",       # don't fail subprocess on HTTP errors
            "-f",            locust_file,
        ]

        print(f"[Load] Starting Locust: {num_users} users at spawn_rate={spawn_rate}/s, {duration_s}s → {config.endpoint_url}")

        # Scale timeout with user count: base 60s + buffer for startup + teardown with high concurrency
        timeout = duration_s + max(60, num_users * 3)

        # Use subprocess.run in a thread executor instead of asyncio.create_subprocess_exec.
        # asyncio.create_subprocess_exec requires ProactorEventLoop on Windows but uvicorn
        # uses SelectorEventLoop — this causes silent failures. subprocess.run in an
        # executor works on all platforms without event-loop compatibility issues.
        loop = asyncio.get_event_loop()

        def _run_locust():
            return subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )

        try:
            proc_result = await loop.run_in_executor(None, _run_locust)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Locust timed out after {timeout}s (duration={duration_s}s, users={num_users})"
            )

        # Always print stderr so server logs show what Locust did
        stderr_text = proc_result.stderr.decode(errors="replace")
        if stderr_text.strip():
            print(f"[Load/Locust stderr]\n{stderr_text[-3000:]}")

        if proc_result.returncode not in (0, 1):   # 1 = some test failures, still valid
            raise RuntimeError(f"Locust subprocess failed (exit {proc_result.returncode}): {stderr_text[-500:]}")

        metrics = _parse_locust_csv(csv_prefix)
        print(
            f"[Load] Locust complete — {metrics['total_requests']} requests, "
            f"p95={metrics['p95_latency_ms']}ms, "
            f"RPS={metrics['requests_per_second']}, "
            f"errors={metrics['error_rate']}%"
        )
        return metrics
