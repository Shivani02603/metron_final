"""
Stage 5: Result Aggregation.
Domain-weighted health score, per-class summaries, persona breakdown, failure drill-down.
Sourced from new metron-backend/app/stage6_results/aggregator.py.
"""

from __future__ import annotations
import statistics
from typing import Any, Dict, List, Optional

from core.config import DOMAIN_WEIGHTS, THRESHOLDS
from core.models import (
    AggregatedReport, ApplicationType, ClassSummary, Conversation,
    MetricResult, Persona, PersonaBreakdown, RunConfig,
)


def aggregate(
    metric_results: List[MetricResult],
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    performance_metrics: Optional[Dict[str, Any]] = None,
    load_metrics: Optional[Dict[str, Any]] = None,
    run_id: str = "",
    project_id: str = "",
    feedback_applied: bool = False,
) -> AggregatedReport:
    """Build the full aggregated report from all evaluation results."""
    domain = config.agent_domain.lower()
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["_default"])

    # ── Per-class summaries ────────────────────────────────────────────────
    test_classes: Dict[str, ClassSummary] = {}

    for test_class in ("functional", "security", "quality", "rag"):
        class_results = [r for r in metric_results if r.superset == test_class]
        if class_results:
            test_classes[test_class] = _summarize(class_results)

    # Performance and load are dict-based (not MetricResult list)
    if performance_metrics:
        total_perf = performance_metrics.get("total_requests", 0)
        passed_perf = performance_metrics.get("successful", 0)
        avg_lat = performance_metrics.get("avg_latency_ms", 0.0)
        p95     = performance_metrics.get("p95_latency_ms", 0.0)
        score   = max(0.0, 1.0 - (p95 / 10000.0))   # latency-based score
        test_classes["performance"] = ClassSummary(
            total=total_perf,
            passed=passed_perf,
            failed=total_perf - passed_perf,
            pass_rate=passed_perf / total_perf if total_perf else 0.0,
            avg_score=score,
            by_metric={"latency": {"p95_ms": p95, "avg_ms": avg_lat}},
        )

    if load_metrics:
        total_load  = load_metrics.get("total_requests", 0)
        passed_load = load_metrics.get("successful", 0)
        p95_load    = load_metrics.get("p95_latency_ms", 0.0)
        score_load  = max(0.0, 1.0 - (load_metrics.get("error_rate", 0) / 100.0))
        test_classes["load"] = ClassSummary(
            total=total_load,
            passed=passed_load,
            failed=total_load - passed_load,
            pass_rate=passed_load / total_load if total_load else 0.0,
            avg_score=score_load,
            by_metric={"load": {"p95_ms": p95_load, "rps": load_metrics.get("requests_per_second", 0)}},
        )

    # ── Health score (domain-weighted) ─────────────────────────────────────
    health_score = _weighted_health(test_classes, weights)

    # ── Persona breakdown ──────────────────────────────────────────────────
    persona_breakdown = _build_persona_breakdown(metric_results, personas)

    # ── Failure drill-down (top 20 worst) ──────────────────────────────────
    failure_drill = _failure_drill_down(metric_results, conversations, 20)

    # ── Totals ─────────────────────────────────────────────────────────────
    total_tests  = len(metric_results)
    total_passed = sum(1 for r in metric_results if r.passed)
    total_failed = total_tests - total_passed

    return AggregatedReport(
        run_id=run_id,
        project_id=project_id,
        application_type=config.application_type,
        domain=config.agent_domain,
        health_score=round(health_score, 4),
        passed=health_score >= THRESHOLDS["health_score_pass"],
        domain_weights=weights,
        test_classes=test_classes,
        persona_breakdown=persona_breakdown,
        failure_drill_down=failure_drill,
        total_tests=total_tests,
        total_passed=total_passed,
        total_failed=total_failed,
        feedback_applied=feedback_applied,
    )


def _summarize(results: List[MetricResult]) -> ClassSummary:
    """Summarize a list of MetricResults into a ClassSummary."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    scores = [r.score for r in results]
    avg    = statistics.mean(scores) if scores else 0.0

    # Per-metric breakdown
    by_metric: Dict[str, Any] = {}
    for r in results:
        mn = r.metric_name
        if mn not in by_metric:
            by_metric[mn] = {"total": 0, "passed": 0, "scores": []}
        by_metric[mn]["total"] += 1
        if r.passed:
            by_metric[mn]["passed"] += 1
        by_metric[mn]["scores"].append(r.score)

    by_metric_clean: Dict[str, Any] = {}
    for mn, data in by_metric.items():
        t = data["total"]
        p = data["passed"]
        s = data["scores"]
        by_metric_clean[mn] = {
            "total":     t,
            "passed":    p,
            "pass_rate": round(p / t, 4) if t else 0.0,
            "avg_score": round(statistics.mean(s), 4) if s else 0.0,
        }

    # Top 5 failures
    top_failures = sorted(results, key=lambda r: r.score)[:5]
    failures_summary = [
        {
            "metric_name": r.metric_name,
            "persona_name": r.persona_name,
            "score": r.score,
            "reason": r.reason[:200],
            "prompt": r.prompt[:150],
        }
        for r in top_failures if not r.passed
    ]

    return ClassSummary(
        total=len(results),
        passed=len(passed),
        failed=len(failed),
        pass_rate=round(len(passed) / len(results), 4) if results else 0.0,
        avg_score=round(avg, 4),
        by_metric=by_metric_clean,
        failures=failures_summary,
    )


def _weighted_health(
    test_classes: Dict[str, ClassSummary],
    weights: Dict[str, float],
) -> float:
    """Compute weighted health score across available test classes."""
    total_weight = 0.0
    weighted_sum = 0.0
    for cls_name, summary in test_classes.items():
        w = weights.get(cls_name, 0.0)
        if w > 0:
            weighted_sum += summary.avg_score * w
            total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _build_persona_breakdown(
    results: List[MetricResult],
    personas: List[Persona],
) -> List[PersonaBreakdown]:
    """Build per-persona pass/fail summary, sorted worst-first."""
    persona_map = {p.persona_id: p for p in personas}
    by_persona: Dict[str, List[MetricResult]] = {}
    for r in results:
        by_persona.setdefault(r.persona_id, []).append(r)

    breakdowns: List[PersonaBreakdown] = []
    for pid, persona_results in by_persona.items():
        persona = persona_map.get(pid)
        passed = [r for r in persona_results if r.passed]
        scores = [r.score for r in persona_results]
        breakdowns.append(PersonaBreakdown(
            persona_id=pid,
            persona_name=persona_results[0].persona_name,
            user_type=persona.user_type if persona else "",
            intent=persona.intent.value if persona else "genuine",
            fishbone=persona.fishbone_dimensions if persona else {},
            total=len(persona_results),
            passed=len(passed),
            failed=len(persona_results) - len(passed),
            avg_score=round(statistics.mean(scores), 4) if scores else 0.0,
            pass_rate=round(len(passed) / len(persona_results), 4) if persona_results else 0.0,
        ))

    return sorted(breakdowns, key=lambda b: b.avg_score)   # worst first


def _failure_drill_down(
    results: List[MetricResult],
    conversations: List[Conversation],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return top N worst failures with full context."""
    conv_map = {c.conversation_id: c for c in conversations}
    failed = [r for r in results if not r.passed]
    failed_sorted = sorted(failed, key=lambda r: r.score)[:limit]

    drill_down = []
    for r in failed_sorted:
        conv = conv_map.get(r.conversation_id)
        turns = []
        if conv:
            for t in conv.turns:
                turns.append({
                    "query": t.query,
                    "response": t.response[:2000],
                    "latency_ms": t.latency_ms,
                })
        drill_down.append({
            "superset":     r.superset,
            "metric_name":  r.metric_name,
            "persona_name": r.persona_name,
            "intent":       r.intent,
            "fishbone":     r.fishbone,
            "score":        r.score,
            "reason":       r.reason,
            "prompt":       r.prompt,
            "response":     r.response[:2000],
            "turns":        turns,
            "vulnerability_found": r.vulnerability_found,
            "pii_detected": r.pii_detected,
        })
    return drill_down
