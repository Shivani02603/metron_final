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
from core.config import THRESHOLDS
from core.models import MetricResult, RunConfig

# ── Generic fallback prompts (used when domain is unrecognised) ────────────────
_GENERIC_PERFORMANCE_PROMPTS = [
    "Hello, what can you help me with?",
    "Please explain your main capabilities briefly.",
    "Can you help me with a complex multi-step problem?",
    "What is the scope of your knowledge?",
    "How do you handle edge cases or unusual requests?",
]

# ── Domain-specific prompt banks ───────────────────────────────────────────────
_DOMAIN_PROMPT_MAP = {
    "email": [
        "Please draft a professional email to schedule a meeting next week.",
        "Write a polite email declining a job offer.",
        "Compose a follow-up email for a proposal sent three days ago.",
        "Draft a concise out-of-office auto-reply for a two-week absence.",
        "Write an email requesting an extension on a project deadline.",
    ],
    "customer_support": [
        "I haven't received my order yet. Can you help?",
        "I'd like to return a product I purchased last week.",
        "My account seems to be locked. How can I regain access?",
        "Can you explain the refund policy for digital purchases?",
        "I'm having trouble connecting to your service. What should I do?",
    ],
    "finance": [
        "What are the current interest rates for a personal loan?",
        "How do I dispute a charge on my credit card statement?",
        "Can you explain how to set up automatic bill payments?",
        "What documents do I need to open a new account?",
        "How do I check my account balance and recent transactions?",
    ],
    "hr": [
        "How many vacation days do I have remaining?",
        "What is the process for requesting parental leave?",
        "Can you explain the performance review schedule?",
        "How do I update my direct deposit information?",
        "What are the guidelines for remote work arrangements?",
    ],
    "healthcare": [
        "How do I schedule an appointment with my doctor?",
        "What are the available options for prescription refills?",
        "Can you explain my insurance coverage for specialist visits?",
        "How do I access my medical records online?",
        "What should I do in a non-emergency medical situation?",
    ],
    "legal": [
        "What is the statute of limitations for contract disputes?",
        "Can you explain the difference between arbitration and mediation?",
        "What are the requirements for a valid contract?",
        "How do I file a complaint with the relevant regulatory authority?",
        "What documentation is typically needed for a property sale?",
    ],
    "ecommerce": [
        "What is the expected delivery time for standard shipping?",
        "How do I track my current order?",
        "Can I modify my order after it has been placed?",
        "What payment methods are accepted?",
        "How does the loyalty points program work?",
    ],
    "travel": [
        "What are the baggage allowance rules for my flight?",
        "How do I upgrade to business class?",
        "Can I change my hotel booking dates without a fee?",
        "What travel insurance options are available?",
        "How do I check my flight status?",
    ],
    "education": [
        "Can you explain the concept of machine learning in simple terms?",
        "What are the key differences between Python and JavaScript?",
        "How do I approach learning a new programming language?",
        "Can you recommend resources for studying data science?",
        "What are the best practices for writing clean code?",
    ],
}


def _get_performance_prompts(config: RunConfig) -> list:
    """
    Return domain-appropriate prompts based on agent_domain and agent_description.
    Falls back to generic prompts when the domain is unrecognised.
    """
    domain      = (getattr(config, "agent_domain",      "") or "").lower().strip()
    description = (getattr(config, "agent_description", "") or "").lower()
    combined    = f"{domain} {description}"

    for key, prompts in _DOMAIN_PROMPT_MAP.items():
        if key in combined:
            return prompts

    return _GENERIC_PERFORMANCE_PROMPTS


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
    prompts = performance_prompts or _get_performance_prompts(config)
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
        request_template=getattr(config, "request_template", None),
        response_trim_marker=getattr(config, "response_trim_marker", None),
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
            elif resp.error is not None:
                # Actual HTTP/connection error — count as failure
                errors += 1
                latencies.append(resp.latency_ms)
            else:
                # HTTP 200 — count as success even if response field extraction failed.
                # Performance measures endpoint speed, not content quality.
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
        "passed":          _percentile(sorted_lat, 95) <= THRESHOLDS["performance_latency_ms"] and errors / total < 0.05,
    }
