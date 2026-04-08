"""
Stage 4d: Performance evaluation.
Built-in async HTTP timing — no external dependencies.
Sourced from existing METRON performance test logic (app_v3.py run_performance_tests).
"""

from __future__ import annotations
import asyncio
import statistics
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


async def evaluate_performance(
    config: RunConfig,
    performance_prompts: Optional[List[str]] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    """
    Send N concurrent requests and measure latency distribution.
    Returns a metrics dict (not MetricResult list — performance is aggregate).
    """
    prompts = performance_prompts or PERFORMANCE_TEST_PROMPTS
    num_requests = min(config.performance_requests, len(prompts) * 4)

    # Cycle through prompts to reach num_requests
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    adapter = ChatbotAdapter(
        endpoint_url=config.endpoint_url,
        request_field=config.request_field,
        response_field=config.response_field,
        auth_type=config.auth_type,
        auth_token=config.auth_token,
        timeout=30,
    )

    # Batch into groups of 5 to avoid overwhelming the server
    BATCH = 5
    latencies: List[float] = []
    errors = 0

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
    n = len(sorted_lat)

    def percentile(data: list, p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    total_time_s = sum(latencies) / 1000.0 or 1.0
    throughput = successful / total_time_s

    return {
        "total_requests":  total,
        "successful":      successful,
        "errors":          errors,
        "error_rate":      round(errors / total * 100, 2),
        "avg_latency_ms":  round(statistics.mean(latencies), 2),
        "min_latency_ms":  round(sorted_lat[0], 2),
        "max_latency_ms":  round(sorted_lat[-1], 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p50_latency_ms":  round(percentile(sorted_lat, 50), 2),
        "p95_latency_ms":  round(percentile(sorted_lat, 95), 2),
        "p99_latency_ms":  round(percentile(sorted_lat, 99), 2),
        "throughput_rps":  round(throughput, 3),
        "passed":          percentile(sorted_lat, 95) <= 5000 and errors / total < 0.05,
    }
