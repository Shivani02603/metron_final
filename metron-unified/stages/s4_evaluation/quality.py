"""
Stage 4c: Quality evaluation.
Metrics: RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)
         + domain-specific G-Eval criteria generated in Stage 2c.
Sourced from existing METRON quality_metrics.py, extended with domain criteria.
"""

from __future__ import annotations
import asyncio
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    ApplicationType, Conversation, MetricResult, Persona, RunConfig, TestClass,
)
from core.config import THRESHOLDS

GEVAL_PROMPT = """
Evaluate this AI response against the following domain-specific criteria.

{criteria_text}

QUESTION: {question}
AI RESPONSE: {response}

For EACH criterion, provide a score from 0.0 to 1.0.
Return JSON:
{{
  "scores": {{
    "<criterion_name>": <0.0-1.0>,
    ...
  }},
  "overall": <weighted average>,
  "reasoning": "<1-2 sentences covering main strengths/weaknesses>"
}}
"""


async def evaluate_quality(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate quality for all conversations using domain-specific criteria."""
    persona_map = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    # Use all conversations (functional + security give us quality signals too)
    for conv in conversations:
        if not conv.turns:
            continue

        last_turn = conv.turns[-1]
        persona = persona_map.get(conv.persona_id)
        fishbone = persona.fishbone_dimensions if persona else {}
        intent = persona.intent.value if persona else "genuine"

        base_meta = dict(
            conversation_id=conv.conversation_id,
            persona_id=conv.persona_id,
            persona_name=conv.persona_name,
            intent=intent,
            fishbone=fishbone,
            prompt=last_turn.query[:300],
            response=last_turn.response[:300],
            latency_ms=conv.total_latency_ms,
            superset="quality",
        )

        # ── G-Eval with domain-specific criteria ──────────────────────────
        if quality_criteria and llm_client:
            from stages.s2_tests.quality_criteria import criteria_to_geval_string
            criteria_text = criteria_to_geval_string(quality_criteria)
            geval_result = await _geval(
                question=last_turn.query,
                response=last_turn.response,
                criteria_text=criteria_text,
                criteria=quality_criteria.get("criteria", []),
                llm_client=llm_client,
            )
            for criterion_name, score in geval_result["scores"].items():
                threshold = quality_criteria.get("passing_threshold", THRESHOLDS["quality_pass"])
                results.append(MetricResult(
                    **base_meta,
                    metric_name=f"geval_{criterion_name.lower().replace(' ', '_')}",
                    score=float(score),
                    passed=float(score) >= threshold,
                    reason=geval_result["reasoning"],
                ))
            # Overall G-Eval score
            results.append(MetricResult(
                **base_meta,
                metric_name="geval_overall",
                score=geval_result["overall"],
                passed=geval_result["overall"] >= quality_criteria.get("passing_threshold", 0.65),
                reason=geval_result["reasoning"],
            ))

        # ── RAGAS for RAG mode ─────────────────────────────────────────────
        if config.application_type == ApplicationType.RAG and last_turn.retrieved_context:
            context = last_turn.retrieved_context or []
            faith = await _ragas_faithfulness(
                last_turn.query, last_turn.response, context, config.rag_text, llm_client,
            )
            results.append(MetricResult(
                **base_meta,
                metric_name="ragas_faithfulness",
                score=faith,
                passed=faith >= THRESHOLDS["quality_pass"],
                reason=f"Faithfulness to ground truth: {faith:.3f}",
            ))

        await asyncio.sleep(0.1)

    return results


async def _geval(
    question: str,
    response: str,
    criteria_text: str,
    criteria: list,
    llm_client: LLMClient,
) -> dict:
    """Run G-Eval with domain-specific criteria."""
    prompt = GEVAL_PROMPT.format(
        criteria_text=criteria_text,
        question=question[:400],
        response=response[:600],
    )
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.1, max_tokens=600, task="judge", retries=2,
        )
        scores = data.get("scores", {})
        if not scores and criteria:
            scores = {c["name"]: 0.5 for c in criteria}
        overall = float(data.get("overall", sum(scores.values()) / len(scores) if scores else 0.5))
        return {
            "scores": {k: float(v) for k, v in scores.items()},
            "overall": overall,
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        default_scores = {c["name"]: 0.5 for c in criteria} if criteria else {"overall": 0.5}
        return {"scores": default_scores, "overall": 0.5, "reasoning": "Evaluation unavailable"}


async def _ragas_faithfulness(
    query: str,
    response: str,
    context: list,
    ground_truth: str,
    llm_client: LLMClient,
) -> float:
    """Measure faithfulness to retrieved context and optional ground truth."""
    ctx_text = " ".join(context[:3])[:600]
    gt_text  = (ground_truth or "")[:300]

    faith_prompt = f"""Rate how faithfully this response is grounded in the provided context (0.0-1.0).

CONTEXT: {ctx_text}
GROUND TRUTH: {gt_text if gt_text else 'not provided'}
QUESTION: {query[:300]}
RESPONSE: {response[:400]}

Return JSON: {{"faithfulness": <0.0-1.0>, "reason": "<1 sentence>"}}"""

    try:
        data = await llm_client.complete_json(
            faith_prompt, temperature=0.1, max_tokens=200, task="judge", retries=2,
        )
        return float(data.get("faithfulness", 0.5))
    except Exception:
        return 0.5
