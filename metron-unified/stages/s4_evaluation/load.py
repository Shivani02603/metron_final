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

from core.config import THRESHOLDS
from core.models import RunConfig

# Import domain-aware prompt generator shared with performance.py
from stages.s4_evaluation.performance import _get_performance_prompts, _GENERIC_PERFORMANCE_PROMPTS

# Keep a short alias for the generic set (used as the bare fallback in preflight)
LOAD_TEST_PROMPTS = _GENERIC_PERFORMANCE_PROMPTS

# ── Locust file template ───────────────────────────────────────────────────────
# Values are injected via json.dumps for safe string encoding.
# When _REQUEST_TEMPLATE is non-empty, it is rendered per-request with {{query}}
# and {{uuid}} substituted; {{conversation_id}} also gets a fresh UUID per request
# (load test has no multi-turn sessions, so each request is its own conversation).

_LOCUST_FILE_TEMPLATE = textwrap.dedent("""\
    import json
    import random
    import uuid as _uuid_mod
    from locust import HttpUser, task, between

    _PROMPTS           = {prompts_json}
    _REQUEST_FIELD     = {request_field_json}
    _RESPONSE_FIELD    = {response_field_json}
    _AUTH_TYPE         = {auth_type_json}
    _AUTH_TOKEN        = {auth_token_json}
    _PATH              = {path_json}
    _REQUEST_TEMPLATE  = {request_template_json}
    _TRIM_MARKER       = {trim_marker_json}


    class AIEndpointUser(HttpUser):
        # Fix 26: tighter wait window gets more requests into the test window
        wait_time = between(0.5, 1.5)

        def on_start(self):
            if _AUTH_TYPE == "bearer" and _AUTH_TOKEN:
                self.client.headers.update({{"Authorization": f"Bearer {{_AUTH_TOKEN}}"}})

        @task
        def send_request(self):
            prompt = random.choice(_PROMPTS)
            if _REQUEST_TEMPLATE:
                body_str = (
                    _REQUEST_TEMPLATE
                    .replace("{{{{query}}}}", prompt)
                    .replace("{{{{uuid}}}}", str(_uuid_mod.uuid4()))
                    .replace("{{{{conversation_id}}}}", str(_uuid_mod.uuid4()))
                )
                payload = json.loads(body_str)
            else:
                payload = {{_REQUEST_FIELD: prompt}}
            with self.client.post(
                _PATH, json=payload, catch_response=True, name="AI Endpoint"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Traverse full dot-notation path (not just first segment)
                        _parts = _RESPONSE_FIELD.split(".") if _RESPONSE_FIELD else []
                        _obj = data
                        _valid = True
                        for _part in _parts:
                            if isinstance(_obj, dict) and _part in _obj:
                                _obj = _obj[_part]
                            elif isinstance(_obj, list) and _part.isdigit():
                                _idx = int(_part)
                                if 0 <= _idx < len(_obj):
                                    _obj = _obj[_idx]
                                else:
                                    _valid = False
                                    break
                            else:
                                _valid = False
                                break
                        if _valid or not _RESPONSE_FIELD:
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

    domain_prompts = _get_performance_prompts(config)
    content = _LOCUST_FILE_TEMPLATE.format(
        prompts_json          = json.dumps(domain_prompts),
        request_field_json    = json.dumps(config.request_field),
        response_field_json   = json.dumps(config.response_field),
        auth_type_json        = json.dumps(config.auth_type),
        auth_token_json       = json.dumps(config.auth_token),
        path_json             = json.dumps(endpoint),
        request_template_json = json.dumps(getattr(config, "request_template", None) or ""),
        trim_marker_json      = json.dumps(getattr(config, "response_trim_marker", None) or ""),
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

                _lat_cap = THRESHOLDS["performance_latency_ms"]
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
                    "passed": p95_ms <= _lat_cap and error_rate < 5.0,
                    "assessment":          _assess_load(error_rate / 100, p95_ms),
                }

    raise ValueError("Locust CSV found but 'Aggregated' row missing.")


def _assess_load(error_rate: float, p95_ms: float) -> str:
    lat_cap = THRESHOLDS["performance_latency_ms"]
    if error_rate < 0.01 and p95_ms < lat_cap * 0.4:
        return "excellent"
    if error_rate < 0.05 and p95_ms < lat_cap:
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
        import uuid as _uuid_mod
        request_template = getattr(config, "request_template", None)
        domain_prompts = _get_performance_prompts(config)
        test_prompt    = domain_prompts[0]
        if request_template:
            body_str = (
                request_template
                .replace("{{query}}", test_prompt)
                .replace("{{uuid}}", str(_uuid_mod.uuid4()))
                .replace("{{conversation_id}}", str(_uuid_mod.uuid4()))
            )
            payload = body_str.encode()
        else:
            payload = json.dumps({config.request_field: test_prompt}).encode()
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

        # Fix 26: ramp-up completes within 20% of test duration (max) to preserve test time.
        # Old formula: num_users // 5 → with 5 users gives spawn_rate=1, wastes 5s of a 30s test.
        # New formula: ramp_time_budget = 20% of duration, spawn_rate = users / budget.
        ramp_time_budget = max(5.0, duration_s * 0.2)
        spawn_rate = max(1, int(num_users / ramp_time_budget))

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
        loop = asyncio.get_running_loop()

        def _run_locust():
            return subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )

        try:
            proc_result = await loop.run_in_executor(None, _run_locust)
        except subprocess.TimeoutExpired:
            # Try to parse whatever CSV Locust wrote before the timeout fired.
            # Locust flushes CSV incrementally so partial results are often available.
            print(f"[Load] Locust timed out after {timeout}s — attempting partial CSV parse...")
            try:
                metrics = _parse_locust_csv(csv_prefix)
                metrics["concurrent_users"] = num_users
                metrics["duration_seconds"] = duration_s
                metrics["warning"] = f"Partial results: Locust timed out after {timeout}s"
                print(f"[Load] Partial results recovered from CSV.")
                return metrics
            except (FileNotFoundError, ValueError):
                raise RuntimeError(
                    f"Locust timed out after {timeout}s with no results written. "
                    f"Try reducing user count or duration. "
                    f"(duration={duration_s}s, users={num_users})"
                )

        # Always print stderr so server logs show what Locust did
        stderr_text = proc_result.stderr.decode(errors="replace")
        if stderr_text.strip():
            print(f"[Load/Locust stderr]\n{stderr_text[-3000:]}")

        if proc_result.returncode not in (0, 1):   # 1 = some test failures, still valid
            raise RuntimeError(f"Locust subprocess failed (exit {proc_result.returncode}): {stderr_text[-500:]}")

        metrics = _parse_locust_csv(csv_prefix)
        # Inject config values that _parse_locust_csv cannot know from the CSV
        metrics["concurrent_users"] = num_users
        metrics["duration_seconds"] = duration_s
        print(
            f"[Load] Locust complete — {metrics['total_requests']} requests, "
            f"p95={metrics['p95_latency_ms']}ms, "
            f"RPS={metrics['requests_per_second']}, "
            f"errors={metrics['error_rate']}%"
        )
        return metrics
