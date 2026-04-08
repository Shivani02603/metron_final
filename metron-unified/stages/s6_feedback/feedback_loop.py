"""
Stage 6: Adaptive Feedback Loop.
Analyzes which personas found failures, generates new targeted persona slots,
runs Stage 2-5 on new personas, and re-aggregates.
One iteration per run. Sourced from new metron-backend/app/stage7_feedback/feedback_loop.py.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

from core.llm_client import LLMClient
from core.models import AggregatedReport, AppProfile, Persona, PersonaFeedback, RunConfig

FAILURE_ANALYSIS_PROMPT = """
Analyze these test failures and identify patterns.

APPLICATION: {domain} {application_type}

FAILING TESTS (worst {count}):
{failures_text}

Return JSON:
{{
  "failure_patterns": ["<pattern 1>", "<pattern 2>"],
  "high_risk_user_types": ["<user type>"],
  "high_risk_topics": ["<topic>"],
  "recommended_new_persona_slots": [
    {{
      "user_type": "<user type>",
      "expertise": "novice" | "intermediate" | "expert",
      "emotional_state": "calm" | "frustrated" | "urgent",
      "intent": "genuine" | "adversarial" | "edge_case",
      "goal_type": "<specific goal targeting the failure pattern>",
      "reason": "<why this slot would expose the pattern>"
    }}
  ]
}}

Rules:
- recommended_new_persona_slots: maximum 3 items
- Each slot must specifically target an identified failure pattern
- Only suggest slots that are meaningfully different from the ones that already ran
"""

EFFECTIVENESS_THRESHOLD = 0.20   # >20% failure rate = effective persona
VARIANT_THRESHOLD       = 0.50   # >50% failure rate = generate variants


def analyze_effectiveness(
    report: AggregatedReport,
    personas: List[Persona],
) -> List[PersonaFeedback]:
    """Determine which personas were effective at finding failures."""
    persona_map = {p.persona_id: p for p in personas}
    feedbacks: List[PersonaFeedback] = []

    for breakdown in report.persona_breakdown:
        persona = persona_map.get(breakdown.persona_id)
        failure_rate = 1.0 - breakdown.pass_rate
        effective = failure_rate > EFFECTIVENESS_THRESHOLD

        if failure_rate > VARIANT_THRESHOLD:
            action = "generate_variants"
        elif failure_rate > EFFECTIVENESS_THRESHOLD:
            action = "strengthen"
        elif breakdown.total >= 3 and failure_rate == 0.0:
            action = "retire"
        else:
            action = "keep"

        # Collect failure patterns from drill-down
        patterns = [
            f["reason"][:100] for f in report.failure_drill_down
            if f["persona_name"] == breakdown.persona_name
        ][:5]

        feedbacks.append(PersonaFeedback(
            persona_id=breakdown.persona_id,
            persona_name=breakdown.persona_name,
            project_id=report.project_id,
            found_failures=breakdown.failed,
            total_runs=breakdown.total,
            failure_rate=round(failure_rate, 4),
            effective=effective,
            failure_patterns=patterns,
            suggested_action=action,
        ))

    return feedbacks


async def generate_new_slots(
    report: AggregatedReport,
    profile: AppProfile,
    llm_client: LLMClient,
) -> List[Dict[str, str]]:
    """LLM analyzes top failures and suggests new persona slots."""
    failures = report.failure_drill_down[:15]
    if not failures:
        return []

    failures_text = "\n".join(
        f"[{f['superset']}] Persona: {f['persona_name']} | Score: {f['score']:.2f} | "
        f"Issue: {f['reason'][:100]} | Prompt: {f['prompt'][:80]}"
        for f in failures
    )

    prompt = FAILURE_ANALYSIS_PROMPT.format(
        domain=profile.domain,
        application_type=profile.application_type.value,
        count=len(failures),
        failures_text=failures_text,
    )

    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.4, max_tokens=1200, task="fast", retries=2,
        )
        slots = data.get("recommended_new_persona_slots", [])
        # Validate
        required = {"user_type", "expertise", "emotional_state", "intent", "goal_type"}
        valid = [s for s in slots[:3] if required.issubset(s.keys())]
        return valid
    except Exception:
        return []


async def run_feedback_loop(
    report: AggregatedReport,
    profile: AppProfile,
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    run_stages_fn: Callable,   # async fn(new_slots, profile, config, llm_client) -> (results, convs, new_personas)
    progress_callback: Optional[Callable] = None,
) -> tuple[AggregatedReport, List[Persona]]:
    """
    Run one feedback iteration:
    1. Analyze effectiveness
    2. Generate new persona slots
    3. Run Stages 2-5 on new personas
    4. Merge results and re-aggregate

    Returns updated (report, all_personas).
    """
    if progress_callback:
        progress_callback(90, "Analyzing test effectiveness…")

    feedbacks = analyze_effectiveness(report, personas)
    effective_count = sum(1 for f in feedbacks if f.effective)

    if effective_count == 0:
        if progress_callback:
            progress_callback(92, "All personas performed similarly — no targeted feedback needed")
        return report, personas

    new_slots = await generate_new_slots(report, profile, llm_client)
    if not new_slots:
        if progress_callback:
            progress_callback(92, "No additional persona slots recommended")
        return report, personas

    if progress_callback:
        progress_callback(92, f"Generating {len(new_slots)} targeted personas for weak areas…")

    # Run Stages 1b, 2, 3, 4, 5 on new slots
    new_results, new_convs, new_personas = await run_stages_fn(
        new_slots, profile, config, llm_client,
    )

    if not new_results:
        return report, personas

    if progress_callback:
        progress_callback(96, "Merging feedback results…")

    # Re-aggregate with combined results
    from stages.s5_aggregation.aggregator import aggregate as _aggregate
    all_personas = personas + new_personas

    # Rebuild combined metric results from both runs
    # The existing report contains partial data; we add new results
    combined_report = _aggregate(
        metric_results=new_results,   # aggregator uses all results internally
        conversations=new_convs,
        personas=all_personas,
        config=config,
        run_id=report.run_id,
        project_id=report.project_id,
        feedback_applied=True,
    )

    # Merge health scores: weighted 70% original, 30% feedback
    merged_health = report.health_score * 0.7 + combined_report.health_score * 0.3
    combined_report.health_score = round(merged_health, 4)
    combined_report.passed = merged_health >= 0.7
    combined_report.persona_breakdown = report.persona_breakdown + combined_report.persona_breakdown
    combined_report.total_tests  += report.total_tests
    combined_report.total_passed += report.total_passed
    combined_report.total_failed += report.total_failed
    # Merge test_classes from original
    for cls_name, cls_summary in report.test_classes.items():
        if cls_name not in combined_report.test_classes:
            combined_report.test_classes[cls_name] = cls_summary

    return combined_report, all_personas
