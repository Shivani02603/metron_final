"""
Stage 4a: Functional evaluation.
Strict tool-only — no heuristics or hardcoded fallback scores.

Metrics:
  - Hallucination    → DeepEval HallucinationMetric  (Azure OpenAI)
  - Answer Relevancy → DeepEval AnswerRelevancyMetric (Azure OpenAI)
  - Usefulness       → DeepEval GEval (custom usefulness criteria)
  - LLM Judge        → domain-specific criteria via configured LLM (always runs)

If a DeepEval metric throws an exception for a specific turn, that metric is
skipped for that conversation — no result is recorded rather than recording a
fabricated score.

Error responses (is_error_response=True) skip all metric scoring.
"""

from __future__ import annotations
import asyncio
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    Conversation, MetricResult, Persona, RunConfig,
)
from core.config import THRESHOLDS
from core.deepeval_azure import make_deepeval_azure_model

LLM_JUDGE_PROMPT = """
Evaluate this AI response on the following criteria.

QUESTION: {question}
AI RESPONSE: {response}
EXPECTED BEHAVIOR: {expected}

Rate the response from 0.0 to 1.0 on each criterion below:
{criteria_text}

Return JSON where each key is the criterion name (lowercase, underscores) and value is 0.0-1.0, plus:
{{
  "<criterion_1>": <0.0-1.0>,
  "<criterion_2>": <0.0-1.0>,
  "overall": <average of all criteria>,
  "reasoning": "<1-2 sentences>"
}}
"""

_DEFAULT_CRITERIA_TEXT = (
    "- relevance: Does it directly address the question?\n"
    "- accuracy: Is the information correct?\n"
    "- helpfulness: Does it help the user achieve their goal?"
)


def _configure_deepeval(llm_provider: str, llm_api_key: str) -> None:
    """Ensure Azure env vars are set so the AzureOpenAI client resolves correctly."""
    try:
        import os
        p = llm_provider.lower()
        if "azure" in p and llm_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = llm_api_key
        elif "openai" in p and llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
    except Exception:
        pass


def _set_azure_env(config: RunConfig) -> None:
    """
    Ensure Azure OpenAI env vars are populated for RAGAS and DeepEval tool calls.
    Reads from RunConfig.llm_api_key when provider is Azure, falls back to existing env.
    Also sets OPENAI_API_VERSION which langchain_openai requires.
    """
    import os
    provider = (config.llm_provider or "").lower()
    if "azure" in provider and config.llm_api_key:
        os.environ.setdefault("AZURE_OPENAI_API_KEY", config.llm_api_key)
    os.environ.setdefault("OPENAI_API_VERSION", os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"))


# ── DeepEval metric helpers (no fallback — let exceptions propagate) ──────────

def _deepeval_hallucination(query: str, response: str, context: list, model) -> tuple[float, str]:
    """
    DeepEval HallucinationMetric.
    score returned is 1.0 - raw_hallucination_score (1.0 = no hallucination).
    Raises on any error — caller decides whether to skip or propagate.
    model: DeepEvalBaseLLM instance (Azure GPT-4o).
    """
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    ctx = context[:3] if context else [query]
    metric = HallucinationMetric(threshold=0.5, model=model)
    test_case = LLMTestCase(input=query, actual_output=response, context=ctx)
    metric.measure(test_case)
    score = round(1.0 - float(metric.score), 4)
    return score, f"DeepEval HallucinationMetric: {score:.3f}"


def _deepeval_answer_relevancy(query: str, response: str, model) -> tuple[float, str]:
    """
    DeepEval AnswerRelevancyMetric.
    Raises on any error — caller decides whether to skip or propagate.
    model: DeepEvalBaseLLM instance (Azure GPT-4o).
    """
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    metric = AnswerRelevancyMetric(threshold=0.5, model=model)
    test_case = LLMTestCase(input=query, actual_output=response)
    metric.measure(test_case)
    score = round(float(metric.score), 4)
    return score, f"DeepEval AnswerRelevancyMetric: {score:.3f}"


def _deepeval_usefulness(query: str, response: str, model) -> tuple[float, str]:
    """
    DeepEval GEval with a usefulness criterion.
    Raises on any error — caller decides whether to skip or propagate.
    model: DeepEvalBaseLLM instance (Azure GPT-4o).
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="Usefulness",
        criteria=(
            "The response is directly useful, actionable, and complete. "
            "It helps the user accomplish their goal without requiring additional "
            "clarification or external resources."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=model,
    )
    test_case = LLMTestCase(input=query, actual_output=response)
    metric.measure(test_case)
    score = round(float(metric.score), 4)
    return score, f"DeepEval GEval usefulness: {score:.3f}"


# ── LLM Judge ─────────────────────────────────────────────────────────────────

async def _llm_judge(
    question: str,
    response: str,
    expected: str,
    llm_client: LLMClient,
    criteria_text: str = _DEFAULT_CRITERIA_TEXT,
) -> dict:
    """LLM-as-judge using domain-specific or default criteria."""
    prompt = LLM_JUDGE_PROMPT.format(
        question=question[:400],
        response=response[:600],
        expected=expected[:300] if expected else "A helpful, accurate response",
        criteria_text=criteria_text,
    )
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.1, max_tokens=500, task="judge", retries=2,
        )
        result = {}
        for k, v in data.items():
            if k == "reasoning":
                result[k] = str(v)
            else:
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    pass
        result.setdefault("overall", 0.5)
        result.setdefault("reasoning", "")
        return result
    except Exception:
        return {"overall": 0.5, "reasoning": "Judge unavailable"}


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_functional(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate all functional conversations. Returns MetricResult list."""
    from core.models import TestClass
    func_convs = [c for c in conversations if c.test_class == TestClass.FUNCTIONAL]
    persona_map = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    _configure_deepeval(config.llm_provider, config.llm_api_key)
    deval_model = make_deepeval_azure_model()

    if quality_criteria and quality_criteria.get("criteria"):
        criteria_lines = [
            f"- {c['name'].lower().replace(' ', '_')}: {c.get('description', c['name'])}"
            for c in quality_criteria["criteria"]
        ]
        criteria_text = "\n".join(criteria_lines)
        pass_threshold = quality_criteria.get("passing_threshold", THRESHOLDS["functional_pass"])
    else:
        criteria_text = _DEFAULT_CRITERIA_TEXT
        pass_threshold = THRESHOLDS["functional_pass"]

    sem = asyncio.Semaphore(5)
    loop = asyncio.get_event_loop()

    async def _eval_one(conv: Conversation) -> List[MetricResult]:
        persona  = persona_map.get(conv.persona_id)
        if not conv.turns:
            return []
        last_turn = conv.turns[-1]
        query    = last_turn.query
        response = last_turn.response
        context  = last_turn.retrieved_context or []
        latency  = conv.total_latency_ms
        fishbone = persona.fishbone_dimensions if persona else {}
        intent   = persona.intent.value if persona else "genuine"
        expected = last_turn.expected_behavior or ""

        base_meta = dict(
            conversation_id=conv.conversation_id,
            persona_id=conv.persona_id,
            persona_name=conv.persona_name,
            intent=intent,
            fishbone=fishbone,
            prompt=query,
            response=response[:500],
            latency_ms=latency,
            superset="functional",
        )
        local: List[MetricResult] = []

        # Error responses — record and skip metric scoring
        if last_turn.is_error_response:
            local.append(MetricResult(
                **base_meta, metric_name="error_response",
                score=0.0, passed=False,
                reason=f"Chatbot returned error: {response[:200]}",
            ))
            return local

        # ── Hallucination ─────────────────────────────────────────────────────
        try:
            hall_score, hall_reason = await loop.run_in_executor(
                None, _deepeval_hallucination, query, response, context, deval_model
            )
            local.append(MetricResult(
                **base_meta, metric_name="hallucination",
                score=hall_score,
                passed=hall_score >= (1.0 - THRESHOLDS["hallucination_max"]),
                reason=hall_reason,
            ))
        except Exception as e:
            print(f"[Hallucination] DeepEval error (conv {conv.conversation_id[:8]}): {e}")

        # ── Answer Relevancy ──────────────────────────────────────────────────
        try:
            rel_score, rel_reason = await loop.run_in_executor(
                None, _deepeval_answer_relevancy, query, response, deval_model
            )
            local.append(MetricResult(
                **base_meta, metric_name="answer_relevancy",
                score=rel_score,
                passed=rel_score >= pass_threshold,
                reason=rel_reason,
            ))
        except Exception as e:
            print(f"[AnswerRelevancy] DeepEval error (conv {conv.conversation_id[:8]}): {e}")

        # ── Usefulness (GEval) ────────────────────────────────────────────────
        try:
            use_score, use_reason = await loop.run_in_executor(
                None, _deepeval_usefulness, query, response, deval_model
            )
            local.append(MetricResult(
                **base_meta, metric_name="usefulness",
                score=use_score,
                passed=use_score >= pass_threshold,
                reason=use_reason,
            ))
        except Exception as e:
            print(f"[Usefulness] DeepEval GEval error (conv {conv.conversation_id[:8]}): {e}")

        # ── LLM Judge — domain-specific criteria ──────────────────────────────
        async with sem:
            judge_result = await _llm_judge(query, response, expected, llm_client, criteria_text)
        reasoning = judge_result.get("reasoning", "")
        skip_keys = {"overall", "reasoning"}
        for crit_key, crit_score in judge_result.items():
            if crit_key in skip_keys:
                continue
            try:
                s = float(crit_score)
            except (TypeError, ValueError):
                continue
            local.append(MetricResult(
                **base_meta, metric_name=f"llm_{crit_key}", score=s,
                passed=s >= pass_threshold, reason=reasoning,
            ))

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in func_convs])
    for batch in batches:
        results.extend(batch)
    return results
