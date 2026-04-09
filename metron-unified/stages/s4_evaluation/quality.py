"""
Stage 4c: Quality evaluation.
Tool-first approach:
  - G-Eval:      DeepEval GEval with domain-specific criteria (token probability scoring)
  - RAG metrics: RAGAS faithfulness + answer_relevancy + context_recall + context_precision
                 (context_recall/precision require ground_truth = expected_behavior from Stage 2)
  - Fallback:    LLM prompt-based scoring only when DeepEval/RAGAS unavailable

DeepEval is configured with the user's LLM key before running any metric.
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


def _is_openai_compatible(llm_provider: str) -> bool:
    p = llm_provider.lower()
    return "azure" in p or "openai" in p


def _configure_deepeval(llm_provider: str, llm_api_key: str):
    """Configure DeepEval — only for OpenAI/Azure (others default to gpt-4o anyway)."""
    try:
        import os
        if _is_openai_compatible(llm_provider) and llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
    except Exception:
        pass


# ── DeepEval GEval ────────────────────────────────────────────────────────────

def _run_deepeval_geval(question: str, response: str, criteria: list) -> Optional[dict]:
    """
    DeepEval GEval — uses chain-of-thought + token probabilities for scoring.
    Returns scores dict or None if DeepEval not available/configured.
    """
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        scores = {}
        for criterion in criteria[:6]:
            name = criterion.get("name", "quality")
            description = criterion.get("description", name)
            try:
                metric = GEval(
                    name=name,
                    criteria=description,
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=0.5,
                )
                test_case = LLMTestCase(input=question, actual_output=response)
                metric.measure(test_case)
                scores[name] = round(float(metric.score), 4)
            except Exception:
                scores[name] = 0.5

        if not scores:
            return None
        overall = round(sum(scores.values()) / len(scores), 4)
        return {"scores": scores, "overall": overall, "method": "deepeval_geval"}
    except ImportError:
        return None
    except Exception:
        return None


async def _geval_llm_fallback(
    question: str,
    response: str,
    criteria_text: str,
    criteria: list,
    llm_client: LLMClient,
) -> dict:
    """LLM-based G-Eval fallback when DeepEval is not installed."""
    prompt = f"""Evaluate this AI response against the following domain-specific criteria.

{criteria_text}

QUESTION: {question[:400]}
AI RESPONSE: {response[:600]}

For EACH criterion, provide a score from 0.0 to 1.0.
Return JSON:
{{
  "scores": {{
    "<criterion_name>": <0.0-1.0>
  }},
  "overall": <average>,
  "reasoning": "<1-2 sentences>"
}}"""
    try:
        data = await llm_client.complete_json(prompt, temperature=0.1, max_tokens=600, task="judge", retries=2)
        scores = data.get("scores", {})
        if not scores and criteria:
            scores = {c["name"]: 0.5 for c in criteria}
        overall = float(data.get("overall", sum(scores.values()) / len(scores) if scores else 0.5))
        return {
            "scores": {k: float(v) for k, v in scores.items()},
            "overall": overall,
            "reasoning": data.get("reasoning", ""),
            "method": "llm_judge",
        }
    except Exception:
        default = {c["name"]: 0.5 for c in criteria} if criteria else {"overall": 0.5}
        return {"scores": default, "overall": 0.5, "reasoning": "Evaluation unavailable", "method": "fallback"}


# ── RAGAS faithfulness + full RAG metrics ────────────────────────────────────

async def _ragas_full(
    query: str,
    response: str,
    context: list,
    ground_truth: str,
    llm_client: LLMClient,
) -> dict[str, tuple[float, str]]:
    """
    Run all available RAGAS metrics.
    - faithfulness: always (no ground_truth needed)
    - answer_relevancy: always
    - context_recall: only when ground_truth available
    - context_precision: only when ground_truth available
    Returns dict of metric_name → (score, method).
    """
    results: dict[str, tuple[float, str]] = {}
    ctx = context[:3]

    loop = asyncio.get_event_loop()

    # Try RAGAS — blocking call, must run in executor
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness as r_faith, answer_relevancy as r_rel
        from datasets import Dataset

        metrics_to_run = [r_faith, r_rel]
        data: dict = {
            "question": [query],
            "answer": [response],
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

        def _run_ragas():
            return ragas_evaluate(ds, metrics=metrics_to_run)

        result = await loop.run_in_executor(None, _run_ragas)

        if "faithfulness" in result:
            results["ragas_faithfulness"] = (round(float(result["faithfulness"]), 4), "ragas")
        if "answer_relevancy" in result:
            results["ragas_answer_relevancy"] = (round(float(result["answer_relevancy"]), 4), "ragas")
        if "context_recall" in result:
            results["ragas_context_recall"] = (round(float(result["context_recall"]), 4), "ragas")
        if "context_precision" in result:
            results["ragas_context_precision"] = (round(float(result["context_precision"]), 4), "ragas")
        return results
    except Exception:
        pass

    # DeepEval FaithfulnessMetric fallback — also blocking, run in executor
    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        def _run_deepeval_faith():
            metric = FaithfulnessMetric(threshold=0.5)
            test_case = LLMTestCase(input=query, actual_output=response, retrieval_context=ctx)
            metric.measure(test_case)
            return round(float(metric.score), 4)

        faith_score = await loop.run_in_executor(None, _run_deepeval_faith)
        results["ragas_faithfulness"] = (faith_score, "deepeval_faithfulness")
        return results
    except Exception:
        pass

    # LLM judge final fallback
    ctx_text = " ".join(ctx)[:500]
    gt_text  = (ground_truth or "")[:200]
    prompt = f"""Rate how faithfully this response is grounded in the context (0.0-1.0).
CONTEXT: {ctx_text}
GROUND TRUTH: {gt_text or 'not provided'}
QUESTION: {query[:300]}
RESPONSE: {response[:400]}
Return JSON: {{"faithfulness": <0.0-1.0>}}"""
    try:
        data = await llm_client.complete_json(prompt, temperature=0.1, max_tokens=150, task="judge", retries=2)
        results["ragas_faithfulness"] = (float(data.get("faithfulness", 0.5)), "llm_judge")
    except Exception:
        results["ragas_faithfulness"] = (0.5, "default")

    return results


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_quality(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate quality using DeepEval GEval + RAGAS (LLM fallback only when tools unavailable)."""

    # Configure DeepEval with user's LLM key
    _configure_deepeval(config.llm_provider, config.llm_api_key)
    use_deepeval_llm = _is_openai_compatible(config.llm_provider)

    persona_map = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    # Pre-compute criteria text once (for LLM fallback)
    criteria_text = None
    criteria_list = []
    if quality_criteria:
        criteria_list = quality_criteria.get("criteria", [])
        if criteria_list:
            try:
                from stages.s2_tests.quality_criteria import criteria_to_geval_string
                criteria_text = criteria_to_geval_string(quality_criteria)
            except Exception:
                criteria_text = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in criteria_list)

    sem = asyncio.Semaphore(5)

    async def _eval_one(conv):
        if not conv.turns:
            return []
        last_turn = conv.turns[-1]

        # Skip error responses
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
        local = []
        threshold = quality_criteria.get("passing_threshold", THRESHOLDS["quality_pass"]) if quality_criteria else THRESHOLDS["quality_pass"]

        # ── G-Eval (DeepEval only for OpenAI/Azure, LLM fallback for others) ──
        if criteria_list:
            loop = asyncio.get_event_loop()
            geval_result = None
            if use_deepeval_llm:
                geval_result = await loop.run_in_executor(
                    None, _run_deepeval_geval, last_turn.query, last_turn.response, criteria_list
                )

            if geval_result is None and llm_client and criteria_text:
                async with sem:
                    geval_result = await _geval_llm_fallback(
                        last_turn.query, last_turn.response,
                        criteria_text, criteria_list, llm_client,
                    )

            if geval_result:
                method = geval_result.get("method", "unknown")
                reasoning = geval_result.get("reasoning", f"Scored via {method}")
                for criterion_name, score in geval_result["scores"].items():
                    local.append(MetricResult(
                        **base_meta,
                        metric_name=f"geval_{criterion_name.lower().replace(' ', '_')}",
                        score=float(score),
                        passed=float(score) >= threshold,
                        reason=f"{reasoning} (method: {method})",
                    ))
                local.append(MetricResult(
                    **base_meta,
                    metric_name="geval_overall",
                    score=geval_result["overall"],
                    passed=geval_result["overall"] >= threshold,
                    reason=f"G-Eval overall (method: {method})",
                ))

        # ── RAGAS for RAG mode ─────────────────────────────────────────────
        if config.application_type == ApplicationType.RAG and last_turn.retrieved_context:
            async with sem:
                rag_metrics = await _ragas_full(
                    last_turn.query, last_turn.response,
                    last_turn.retrieved_context,
                    expected,   # ground_truth — unlocks context_recall + context_precision
                    llm_client,
                )
            for metric_name, (score, method) in rag_metrics.items():
                local.append(MetricResult(
                    **base_meta,
                    metric_name=metric_name,
                    score=score,
                    passed=score >= THRESHOLDS["quality_pass"],
                    reason=f"{metric_name.replace('_', ' ').title()}: {score:.3f} (method: {method})",
                ))

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in conversations])
    for batch in batches:
        results.extend(batch)
    return results
