"""
Stage 4d: Performance evaluation.
Built-in async HTTP timing — no external dependencies.

Fix 6: throughput uses wall-clock time (time.monotonic), not sum(latencies).
Fix 7: percentile uses linear interpolation so p95 != max for small samples.
Fix 27: unique nonce appended to each prompt to prevent response caching.
"""

from __future__ import annotations
import asyncio
import statistics
import time
from typing import Any, Dict, List, Optional

from core.adapters.chatbot import ChatbotAdapter
from core.models import MetricResult, RunConfig

PERFORMANCE_TEST_PROMPTS = [
    "Hello, what can you help me with?",
    "Please explain your main capabilities briefly.",
    "Can you help me with a complex multi-step problem?",
    "What is the scope of your knowledge?",
    "How do you handle edge cases or unusual requests?",
]


def _percentile(data: list, p: float) -> float:
    """
    Fix 7: Standard linear interpolation percentile.
    With 20 samples, p95 correctly returns ~19th value, not the max.
    """
    if not data:
        return 0.0
    n = len(data)
    if n == 1:
        return data[0]
    idx = (p / 100) * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return data[lo] + (idx - lo) * (data[hi] - data[lo])


async def evaluate_performance(
    config: RunConfig,
    performance_prompts: Optional[List[str]] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    """
    Send N concurrent requests and measure latency distribution.
    Returns a metrics dict (not MetricResult list — performance is aggregate).

    Fix 6: throughput = successful / wall_clock_seconds (not sum of latencies).
    Fix 7: percentile uses linear interpolation (no more p95 == max).
    Fix 27: each prompt gets a unique nonce to prevent endpoint response caching.
    """
    prompts = performance_prompts or PERFORMANCE_TEST_PROMPTS
    num_requests = min(config.performance_requests, len(prompts) * 4)

    # Cycle through prompts to reach num_requests
    # Fix 27: add unique nonce per request to prevent caching
    test_prompts = [
        f"{prompts[i % len(prompts)]} [ref:{i}]"
        for i in range(num_requests)
    ]

    adapter = ChatbotAdapter(
        endpoint_url=config.endpoint_url,
        request_field=config.request_field,
        response_field=config.response_field,
        auth_type=config.auth_type,
        auth_token=config.auth_token,
        timeout=30,
    )

    BATCH = 5
    latencies: List[float] = []
    errors = 0

    # Fix 6: record wall-clock start before any requests
    wall_start = time.monotonic()

    for i in range(0, len(test_prompts), BATCH):
        batch = test_prompts[i:i + BATCH]
        tasks = [adapter.send(p) for p in batch]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, Exception):
                errors += 1
            elif not resp.ok:
                errors += 1
                latencies.append(resp.latency_ms)
            else:
                latencies.append(resp.latency_ms)
        await asyncio.sleep(0.2)

    # Fix 6: use actual elapsed wall-clock time for throughput
    wall_elapsed_s = time.monotonic() - wall_start or 1.0

    total = len(test_prompts)
    successful = total - errors

    if not latencies:
        return {
            "total_requests": total, "successful": 0, "errors": total,
            "error_rate": 100.0, "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0, "max_latency_ms": 0.0,
            "median_latency_ms": 0.0, "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0, "p99_latency_ms": 0.0, "throughput_rps": 0.0,
        }

    sorted_lat = sorted(latencies)

    # Fix 6: correct throughput formula
    throughput = successful / wall_elapsed_s

    return {
        "total_requests":  total,
        "successful":      successful,
        "errors":          errors,
        "error_rate":      round(errors / total * 100, 2),
        "avg_latency_ms":  round(statistics.mean(latencies), 2),
        "min_latency_ms":  round(sorted_lat[0], 2),
        "max_latency_ms":  round(sorted_lat[-1], 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p50_latency_ms":  round(_percentile(sorted_lat, 50), 2),
        "p95_latency_ms":  round(_percentile(sorted_lat, 95), 2),
        "p99_latency_ms":  round(_percentile(sorted_lat, 99), 2),
        "throughput_rps":  round(throughput, 3),
        "passed":          _percentile(sorted_lat, 95) <= 5000 and errors / total < 0.05,
    }
