"""
Stage 4e: Load test evaluation.
Built-in async concurrent users — NOT Locust subprocess (avoids gevent conflicts with uvicorn).
Sourced from existing METRON built-in async load tester (app_v3.py run_load_tests).
"""

from __future__ import annotations
import asyncio
import statistics
import time
from typing import Any, Dict, List

from core.adapters.chatbot import ChatbotAdapter
from core.models import RunConfig

LOAD_TEST_PROMPTS = [
    "Hello, can you help me?",
    "What are your capabilities?",
    "I need assistance with a task.",
]


async def evaluate_load(config: RunConfig) -> Dict[str, Any]:
    """
    Simulate concurrent users sending requests.
    Returns aggregate load metrics dict.
    """
    num_users    = config.load_concurrent_users
    duration_s   = config.load_duration_seconds
    req_per_user = max(2, duration_s // 10)   # ~1 req per 10s per user

    adapter = ChatbotAdapter(
        endpoint_url=config.endpoint_url,
        request_field=config.request_field,
        response_field=config.response_field,
        auth_type=config.auth_type,
        auth_token=config.auth_token,
        timeout=30,
    )

    start_time = time.monotonic()
    all_latencies: List[float] = []
    errors = 0
    total_requests = num_users * req_per_user

    async def user_session(user_idx: int):
        nonlocal errors
        for req_idx in range(req_per_user):
            prompt = LOAD_TEST_PROMPTS[req_idx % len(LOAD_TEST_PROMPTS)]
            resp = await adapter.send(prompt)
            if resp.ok:
                all_latencies.append(resp.latency_ms)
            else:
                errors += 1
                all_latencies.append(resp.latency_ms)
            await asyncio.sleep(0.1)

    # Run all users concurrently
    tasks = [user_session(i) for i in range(num_users)]
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - start_time
    successful = total_requests - errors

    if not all_latencies:
        return {
            "concurrent_users": num_users,
            "duration_seconds": elapsed,
            "total_requests": total_requests,
            "successful": 0, "errors": total_requests,
            "error_rate": 100.0, "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0, "requests_per_second": 0.0,
            "passed": False,
        }

    sorted_lat = sorted(all_latencies)

    def percentile(data: list, p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    rps = successful / elapsed if elapsed > 0 else 0.0
    p95 = percentile(sorted_lat, 95)

    return {
        "concurrent_users":    num_users,
        "duration_seconds":    round(elapsed, 2),
        "total_requests":      total_requests,
        "successful":          successful,
        "errors":              errors,
        "error_rate":          round(errors / total_requests * 100, 2),
        "avg_latency_ms":      round(statistics.mean(all_latencies), 2),
        "p95_latency_ms":      round(p95, 2),
        "requests_per_second": round(rps, 3),
        "passed":              p95 <= 5000 and errors / total_requests < 0.05,
        "assessment":          _assess_load(errors / total_requests, p95),
    }


def _assess_load(error_rate: float, p95_ms: float) -> str:
    if error_rate < 0.01 and p95_ms < 2000:
        return "excellent"
    if error_rate < 0.05 and p95_ms < 5000:
        return "acceptable"
    if error_rate < 0.10:
        return "degraded"
    return "critical"
