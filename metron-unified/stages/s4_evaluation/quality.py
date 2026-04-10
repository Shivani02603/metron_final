"""
Stage 4c: Quality evaluation.
Strict tool-only — no LLM fallback scores.

Metrics:
  - GEval (domain-specific criteria) → DeepEval GEval (Azure OpenAI)
                                        Skipped per-conversation if DeepEval throws.
  - RAGAS Faithfulness               → RAGAS (RAG mode only)
  - RAGAS Answer Relevancy           → RAGAS (RAG mode only)
  - RAGAS Context Recall             → RAGAS (RAG mode + ground truth required)
  - RAGAS Context Precision          → RAGAS (RAG mode + ground truth required)

Error responses (is_error_response=True) are skipped.
"""

from __future__ import annotations
import asyncio
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    ApplicationType, Conversation, MetricResult, Persona, RunConfig, TestClass,
)
from core.config import THRESHOLDS
from core.deepeval_azure import make_deepeval_azure_model


# ── DeepEval GEval ────────────────────────────────────────────────────────────

def _run_deepeval_geval(question: str, response: str, criteria: list, model) -> dict:
    """
    DeepEval GEval with domain-specific criteria.
    Raises on failure — caller skips metric for this conversation.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    scores = {}
    for criterion in criteria[:6]:
        name        = criterion.get("name", "quality")
        description = criterion.get("description", name)
        metric = GEval(
            name=name,
            criteria=description,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=model,
        )
        test_case = LLMTestCase(input=question, actual_output=response)
        metric.measure(test_case)
        scores[name] = round(float(metric.score), 4)

    if not scores:
        raise RuntimeError("GEval returned no scores")

    overall = round(sum(scores.values()) / len(scores), 4)
    return {"scores": scores, "overall": overall, "method": "deepeval_geval"}


# ── RAGAS ─────────────────────────────────────────────────────────────────────

async def _ragas_full(
    query: str,
    response: str,
    context: list,
    ground_truth: str,
) -> dict[str, tuple[float, str]]:
    """
    Run all available RAGAS metrics.
    - faithfulness + answer_relevancy: always (no ground_truth needed)
    - context_recall + context_precision: only when ground_truth present
    Returns dict of metric_name → (score, method).
    Raises on failure — caller handles.
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness as r_faith, answer_relevancy as r_rel
    from datasets import Dataset   # type: ignore

    ctx = context[:3]
    metrics_to_run = [r_faith, r_rel]
    data: dict = {
        "question": [query],
        "answer":   [response],
        "contexts": [ctx],
    }

    has_gt = bool(ground_truth and ground_truth.strip())
    if has_gt:
        try:
            from ragas.metrics import context_recall as r_recall, context_precision as r_prec
            metrics_to_run += [r_recall, r_prec]
            data["ground_truth"] = [ground_truth]
        except ImportError:
            pass

    ds = Dataset.from_dict(data)
    loop = asyncio.get_event_loop()

    def _run_ragas():
        return ragas_evaluate(ds, metrics=metrics_to_run)

    result = await loop.run_in_executor(None, _run_ragas)

    results: dict[str, tuple[float, str]] = {}
    if "faithfulness" in result:
        results["ragas_faithfulness"]       = (round(float(result["faithfulness"]), 4),       "ragas")
    if "answer_relevancy" in result:
        results["ragas_answer_relevancy"]   = (round(float(result["answer_relevancy"]), 4),   "ragas")
    if "context_recall" in result:
        results["ragas_context_recall"]     = (round(float(result["context_recall"]), 4),     "ragas")
    if "context_precision" in result:
        results["ragas_context_precision"]  = (round(float(result["context_precision"]), 4),  "ragas")
    return results


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_quality(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate quality using DeepEval GEval + RAGAS (tool-only, no LLM fallback)."""

    deval_model = make_deepeval_azure_model()

    persona_map  = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    criteria_list: list = []
    if quality_criteria:
        criteria_list = quality_criteria.get("criteria", [])

    threshold = (
        quality_criteria.get("passing_threshold", THRESHOLDS["quality_pass"])
        if quality_criteria else THRESHOLDS["quality_pass"]
    )

    sem  = asyncio.Semaphore(5)
    loop = asyncio.get_event_loop()

    async def _eval_one(conv: Conversation) -> List[MetricResult]:
        if not conv.turns:
            return []
        last_turn = conv.turns[-1]

        if last_turn.is_error_response:
            return []

        persona  = persona_map.get(conv.persona_id)
        fishbone = persona.fishbone_dimensions if persona else {}
        intent   = persona.intent.value if persona else "genuine"
        expected = last_turn.expected_behavior or ""

        base_meta = dict(
            conversation_id=conv.conversation_id,
            persona_id=conv.persona_id,
            persona_name=conv.persona_name,
            intent=intent, fishbone=fishbone,
            prompt=last_turn.query[:300],
            response=last_turn.response[:300],
            latency_ms=conv.total_latency_ms,
            superset="quality",
        )
        local: List[MetricResult] = []

        # ── GEval (domain-specific criteria) ─────────────────────────────────
        if criteria_list:
            try:
                geval_result = await loop.run_in_executor(
                    None, _run_deepeval_geval,
                    last_turn.query, last_turn.response, criteria_list, deval_model
                )
                method    = geval_result.get("method", "deepeval_geval")
                for criterion_name, score in geval_result["scores"].items():
                    local.append(MetricResult(
                        **base_meta,
                        metric_name=f"geval_{criterion_name.lower().replace(' ', '_')}",
                        score=float(score),
                        passed=float(score) >= threshold,
                        reason=f"DeepEval GEval {criterion_name} (method: {method})",
                    ))
                local.append(MetricResult(
                    **base_meta,
                    metric_name="geval_overall",
                    score=geval_result["overall"],
                    passed=geval_result["overall"] >= threshold,
                    reason=f"GEval overall (method: {method})",
                ))
            except Exception as e:
                print(f"[GEval] DeepEval error (conv {conv.conversation_id[:8]}): {e}")

        # ── RAGAS (RAG mode only) ─────────────────────────────────────────────
        if config.application_type == ApplicationType.RAG and last_turn.retrieved_context:
            async with sem:
                try:
                    rag_metrics = await _ragas_full(
                        last_turn.query,
                        last_turn.response,
                        last_turn.retrieved_context,
                        expected,
                    )
                    for metric_name, (score, method) in rag_metrics.items():
                        local.append(MetricResult(
                            **base_meta,
                            metric_name=metric_name,
                            score=score,
                            passed=score >= THRESHOLDS["quality_pass"],
                            reason=f"{metric_name.replace('_', ' ').title()}: {score:.3f} (method: {method})",
                        ))
                except Exception as e:
                    print(f"[RAGAS] Error (conv {conv.conversation_id[:8]}): {e}")

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in conversations])
    for batch in batches:
        results.extend(batch)
    return results
