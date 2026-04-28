"""
Stage 5: Result Aggregation.
Domain-weighted health score, per-class summaries, persona breakdown, failure drill-down.

Fix 12: safety_passive class added — passive PII/toxicity/bias checks on functional
        conversations are now reported separately from active security attack results.
Fix 34: agent_name parameter added to aggregate(); set on AggregatedReport model.
Fix 36: Skipped-aware pass rate: pass_rate = passed / (total - skipped).
        ClassSummary.skipped surfaced; evaluation_warnings added when >50% of a metric's
        evaluations were skipped (API errors, rate limits) — prevents fake 100% pass rates.
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
    agent_name: str = "",  # Fix 34
) -> AggregatedReport:
    """Build the full aggregated report from all evaluation results."""
    domain = config.agent_domain.lower()
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["_default"])

    # ── Per-class summaries ────────────────────────────────────────────────────
    test_classes: Dict[str, ClassSummary] = {}

    # Fix 12: include safety_passive as an explicit class (informational, 0.0 health weight)
    for test_class in ("functional", "security", "safety_passive", "quality", "rag"):
        class_results = [r for r in metric_results if r.superset == test_class]
        if class_results:
            test_classes[test_class] = _summarize(class_results)

    # Performance and load are dict-based (not MetricResult list)
    if performance_metrics:
        total_perf  = performance_metrics.get("total_requests", 0)
        passed_perf = performance_metrics.get("successful", 0)
        avg_lat    = performance_metrics.get("avg_latency_ms", 0.0)
        p95        = performance_metrics.get("p95_latency_ms", 0.0)
        perf_cap   = float(THRESHOLDS["performance_latency_ms"])
        score      = max(0.0, 1.0 - (p95 / perf_cap))
        test_classes["performance"] = ClassSummary(
            total=total_perf,
            passed=passed_perf,
            failed=total_perf - passed_perf,
            pass_rate=round(passed_perf / total_perf, 4) if total_perf else 0.0,
            avg_score=score,
            by_metric={"latency": {"p95_ms": p95, "avg_ms": avg_lat}},
        )

    if load_metrics:
        total_load     = load_metrics.get("total_requests", 0)
        passed_load    = load_metrics.get("successful", 0)
        p95_load       = load_metrics.get("p95_latency_ms", 0.0)
        load_error_rate = load_metrics.get("error_rate", 0.0)
        load_cap_ms    = float(THRESHOLDS["performance_latency_ms"])
        # Blended score: 60% error-rate health + 40% latency health.
        # Previously only error_rate was used — a slow-but-reliable endpoint
        # (0% errors, 30s p95) incorrectly scored 1.0 (perfect).
        error_health   = max(0.0, 1.0 - (load_error_rate / 100.0))
        latency_health = max(0.0, 1.0 - (p95_load / load_cap_ms))
        score_load     = round(0.6 * error_health + 0.4 * latency_health, 4)
        test_classes["load"] = ClassSummary(
            total=total_load,
            passed=passed_load,
            failed=total_load - passed_load,
            pass_rate=round(passed_load / total_load, 4) if total_load else 0.0,
            avg_score=score_load,
            by_metric={"load": {"p95_ms": p95_load, "rps": load_metrics.get("requests_per_second", 0)}},
        )

    # ── Health score (domain-weighted) ─────────────────────────────────────────
    health_score = _weighted_health(test_classes, weights)

    # ── Persona breakdown ──────────────────────────────────────────────────────
    persona_breakdown = _build_persona_breakdown(metric_results, personas)

    # ── Failure drill-down (top 20 worst) ──────────────────────────────────────
    failure_drill = _failure_drill_down(metric_results, conversations, 20)

    # ── Totals (Fix 36: exclude skipped from counts) ───────────────────────────
    non_skipped    = [r for r in metric_results if not r.skipped]
    total_skipped  = sum(1 for r in metric_results if r.skipped)
    total_tests    = len(non_skipped)
    total_passed   = sum(1 for r in non_skipped if r.passed)
    total_failed   = total_tests - total_passed

    # ── Cross-class evaluation warnings (Fix 36) ──────────────────────────────
    evaluation_warnings: List[str] = []
    for cls_name, summary in test_classes.items():
        evaluation_warnings.extend(summary.evaluation_warnings)

    # Fix 34: derive agent_name fallback from config if not explicitly provided
    effective_agent_name = (
        agent_name
        or getattr(config, "agent_name", "")
        or f"{config.agent_domain.capitalize()} Agent"
    )

    return AggregatedReport(
        run_id=run_id,
        project_id=project_id,
        agent_name=effective_agent_name,   # Fix 34
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
        total_skipped=total_skipped,
        feedback_applied=feedback_applied,
        evaluation_warnings=evaluation_warnings,
    )


def _summarize(results: List[MetricResult]) -> ClassSummary:
    """
    Summarize a list of MetricResults into a ClassSummary.

    Fix 36: skipped metrics excluded from pass-rate denominator.
    evaluation_warnings generated when >50% of any metric's evaluations were skipped.
    """
    skipped_results     = [r for r in results if r.skipped]
    non_skipped_results = [r for r in results if not r.skipped]
    passed              = [r for r in non_skipped_results if r.passed]
    failed              = [r for r in non_skipped_results if not r.passed]
    scores              = [r.score for r in non_skipped_results]
    avg                 = statistics.mean(scores) if scores else 0.0

    # Fix 36: honest denominator = non-skipped only
    effective_total = len(non_skipped_results)
    pass_rate = round(len(passed) / effective_total, 4) if effective_total > 0 else 0.0

    # Per-metric breakdown (with skipped tracking per metric)
    by_metric: Dict[str, Any] = {}
    for r in results:
        mn = r.metric_name
        if mn not in by_metric:
            by_metric[mn] = {"total": 0, "passed": 0, "skipped": 0, "scores": []}
        by_metric[mn]["total"] += 1
        if r.skipped:
            by_metric[mn]["skipped"] += 1
        elif r.passed:
            by_metric[mn]["passed"] += 1
        if not r.skipped:
            by_metric[mn]["scores"].append(r.score)

    by_metric_clean: Dict[str, Any] = {}
    evaluation_warnings: List[str] = []

    for mn, data in by_metric.items():
        t       = data["total"]
        p       = data["passed"]
        sk      = data["skipped"]
        s       = data["scores"]
        eff_t   = t - sk   # effective total (non-skipped)
        pr      = round(p / eff_t, 4) if eff_t > 0 else 0.0

        by_metric_clean[mn] = {
            "total":     t,
            "passed":    p,
            "skipped":   sk,
            "pass_rate": pr,
            "avg_score": round(statistics.mean(s), 4) if s else 0.0,
        }

        # Fix 36: warn when >50% of this metric's evaluations were skipped
        if t > 0 and sk / t > 0.5:
            evaluation_warnings.append(
                f"Over 50% of '{mn}' evaluations were skipped due to API errors "
                f"({sk}/{t}) — results unreliable. Check Azure rate limits."
            )

    # Top 5 failures (non-skipped only)
    top_failures = sorted(non_skipped_results, key=lambda r: r.score)[:5]
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
        skipped=len(skipped_results),
        pass_rate=pass_rate,
        avg_score=round(avg, 4),
        by_metric=by_metric_clean,
        failures=failures_summary,
        evaluation_warnings=evaluation_warnings,
    )


def _weighted_health(
    test_classes: Dict[str, ClassSummary],
    weights: Dict[str, float],
) -> float:
    """
    Compute weighted health score across available test classes.
    safety_passive is informational (weight=0.0) — does not affect health score.
    """
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
    """Build per-persona pass/fail summary, sorted worst-first. Excludes skipped metrics."""
    persona_map = {p.persona_id: p for p in personas}
    by_persona: Dict[str, List[MetricResult]] = {}
    for r in results:
        if not r.skipped:   # exclude skipped metrics from persona stats
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
    """Return top N worst failures with full context. Excludes skipped metrics."""
    conv_map    = {c.conversation_id: c for c in conversations}
    failed      = [r for r in results if not r.passed and not r.skipped]
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
