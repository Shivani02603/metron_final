"""
Stage 4a: Functional evaluation.
Tool-first metrics:
  - ROUGE-L F1          — rouge-score library (no LLM), ref = expected_behavior
  - BERTScore F1        — bert-score library (no LLM), ref = expected_behavior
  - Hallucination       — DeepEval HallucinationMetric
  - Answer Relevancy    — DeepEval AnswerRelevancyMetric
  - LLM Judge           — domain-specific criteria (LLM, only for nuanced scoring)
  - RAG Faithfulness    — RAGAS (no LLM, for RAG mode)

Error responses ([Error: HTTP ...]) are detected via ConversationTurn.is_error_response
and skipped from metric scoring (recorded as failed with score=0).
"""

from __future__ import annotations
import asyncio
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    ApplicationType, Conversation, MetricResult, Persona, RunConfig,
)
from core.config import THRESHOLDS

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

_DEFAULT_CRITERIA_TEXT = "- relevance: Does it directly address the question?\n- accuracy: Is the information correct?\n- helpfulness: Does it help the user achieve their goal?"


def _is_openai_compatible(llm_provider: str) -> bool:
    """Returns True only for providers DeepEval can natively use (OpenAI/Azure).
    For Groq/Gemini/NVIDIA, DeepEval defaults to gpt-4o anyway, causing slow
    external API calls we cannot control — so we skip LLM-based DeepEval metrics."""
    p = llm_provider.lower()
    return "azure" in p or "openai" in p


def _configure_deepeval(llm_provider: str, llm_api_key: str):
    """Configure DeepEval to use the user's LLM key instead of defaulting to OpenAI."""
    try:
        import os
        provider_lower = llm_provider.lower()
        if ("azure" in provider_lower or "openai" in provider_lower) and llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
    except Exception:
        pass


async def evaluate_functional(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate all functional conversations and return MetricResult list."""
    from core.models import TestClass
    func_convs = [c for c in conversations if c.test_class == TestClass.FUNCTIONAL]
    persona_map = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    # Configure DeepEval with the user's LLM key once before running metrics
    _configure_deepeval(config.llm_provider, config.llm_api_key)

    # Build criteria text from quality_criteria (domain-specific) or fallback to defaults
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
    use_deepeval_llm = _is_openai_compatible(config.llm_provider)

    async def _eval_one(conv):
        persona = persona_map.get(conv.persona_id)
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
        local = []

        # ── Skip metric scoring for error responses ────────────────────────
        if last_turn.is_error_response:
            local.append(MetricResult(
                **base_meta, metric_name="error_response",
                score=0.0, passed=False,
                reason=f"Chatbot returned error: {response[:200]}",
            ))
            return local

        # ── ROUGE-L (rouge-score library, no LLM) ──────────────────────────
        # Use expected_behavior as reference when available; skip if empty
        if expected:
            rouge_score = _rouge_l(expected, response)
            local.append(MetricResult(
                **base_meta, metric_name="rouge_l", score=rouge_score,
                passed=rouge_score >= THRESHOLDS["rouge_l_min"],
                reason=f"ROUGE-L F1: {rouge_score:.3f} (ref: expected_behavior)",
            ))

        # ── BERTScore (bert-score library, no LLM) ─────────────────────────
        if expected:
            bert_score = _bert_score_approx(expected, response)
            local.append(MetricResult(
                **base_meta, metric_name="bert_score_f1", score=bert_score,
                passed=bert_score >= THRESHOLDS["bert_score_min"],
                reason=f"BERTScore F1: {bert_score:.3f} (ref: expected_behavior)",
            ))

        # ── DeepEval: Hallucination + Answer Relevancy ──────────────────────
        # Only use LLM-based DeepEval metrics for OpenAI/Azure — for other
        # providers DeepEval silently falls back to gpt-4o, causing slow
        # external calls. We go straight to fast heuristics for other providers.
        loop = asyncio.get_event_loop()
        if use_deepeval_llm:
            hall_score, hall_reason = await loop.run_in_executor(
                None, _deepeval_hallucination, query, response, context
            )
            rel_score, rel_reason = await loop.run_in_executor(
                None, _deepeval_answer_relevancy, query, response
            )
        else:
            hall_score, hall_reason = _hallucination_heuristic(query, response, context)
            rel_score, rel_reason = _relevancy_heuristic(query, response)

        local.append(MetricResult(
            **base_meta, metric_name="hallucination",
            score=hall_score,
            passed=hall_score >= (1.0 - THRESHOLDS["hallucination_max"]),
            reason=hall_reason,
        ))
        local.append(MetricResult(
            **base_meta, metric_name="answer_relevancy",
            score=rel_score,
            passed=rel_score >= pass_threshold,
            reason=rel_reason,
        ))

        # ── LLM Judge — domain-specific criteria ───────────────────────────
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

        # ── RAG faithfulness ───────────────────────────────────────────────
        if context and config.application_type == ApplicationType.RAG:
            async with sem:
                faith = await _faithfulness(query, response, context, llm_client)
            local.append(MetricResult(
                **base_meta, metric_name="faithfulness", score=faith,
                passed=faith >= THRESHOLDS["functional_pass"],
                reason=f"Faithfulness to retrieved context: {faith:.3f}",
            ))

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in func_convs])
    for batch in batches:
        results.extend(batch)
    return results


# ── Metric helpers ─────────────────────────────────────────────────────────

def _rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 using longest common subsequence."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(reference, hypothesis)
        return round(score["rougeL"].fmeasure, 4)
    except ImportError:
        return _rouge_l_fallback(reference, hypothesis)


def _rouge_l_fallback(reference: str, hypothesis: str) -> float:
    """Pure-Python LCS-based ROUGE-L (no external library)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _bert_score_approx(reference: str, hypothesis: str) -> float:
    """BERTScore using bert-score library; token overlap fallback."""
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score([hypothesis], [reference], lang="en", verbose=False)
        return round(float(F1[0]), 4)
    except ImportError:
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        if not ref_set or not hyp_set:
            return 0.0
        overlap = ref_set & hyp_set
        precision = len(overlap) / len(hyp_set)
        recall    = len(overlap) / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)


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


def _hallucination_heuristic(query: str, response: str, context: list) -> tuple[float, str]:
    """Fast heuristic fallback (no external calls)."""
    ref_words = set(query.lower().split())
    hyp_words = set(response.lower().split())
    overlap = len(ref_words & hyp_words) / max(len(hyp_words), 1)
    score = round(min(1.0, overlap * 2), 4)
    return score, f"Hallucination proxy (overlap heuristic): {score:.3f}"


def _relevancy_heuristic(query: str, response: str) -> tuple[float, str]:
    """Fast heuristic fallback (no external calls)."""
    q_words = set(query.lower().split())
    r_words = set(response.lower().split())
    if not q_words or not r_words:
        return 0.5, "Answer relevancy fallback: 0.500"
    overlap = len(q_words & r_words) / len(q_words)
    score = round(min(1.0, overlap * 1.5), 4)
    return score, f"Answer relevancy (token overlap heuristic): {score:.3f}"


def _deepeval_hallucination(query: str, response: str, context: list) -> tuple[float, str]:
    """
    DeepEval HallucinationMetric.
    Falls back to overlap heuristic when DeepEval not installed or not configured.
    Returns (score 0-1, reason). 1.0 = no hallucination.
    """
    try:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase
        ctx = context[:3] if context else [query]
        metric = HallucinationMetric(threshold=0.5)
        test_case = LLMTestCase(input=query, actual_output=response, context=ctx)
        metric.measure(test_case)
        score = round(1.0 - float(metric.score), 4)
        return score, f"DeepEval hallucination: {score:.3f}"
    except ImportError:
        pass
    except Exception:
        pass
    # Heuristic fallback
    ref_words = set(query.lower().split())
    hyp_words = set(response.lower().split())
    overlap = len(ref_words & hyp_words) / max(len(hyp_words), 1)
    score = round(min(1.0, overlap * 2), 4)
    return score, f"Hallucination proxy (overlap): {score:.3f}"


def _deepeval_answer_relevancy(query: str, response: str) -> tuple[float, str]:
    """
    DeepEval AnswerRelevancyMetric.
    Falls back to token overlap when DeepEval not installed or not configured.
    """
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        metric = AnswerRelevancyMetric(threshold=0.5)
        test_case = LLMTestCase(input=query, actual_output=response)
        metric.measure(test_case)
        score = round(float(metric.score), 4)
        return score, f"DeepEval answer relevancy: {score:.3f}"
    except ImportError:
        pass
    except Exception:
        pass
    # Token overlap fallback
    q_words = set(query.lower().split())
    r_words  = set(response.lower().split())
    if not q_words or not r_words:
        return 0.5, "Answer relevancy fallback: 0.500"
    overlap = len(q_words & r_words) / len(q_words)
    score = round(min(1.0, overlap * 1.5), 4)
    return score, f"Answer relevancy (token overlap): {score:.3f}"


async def _faithfulness(
    query: str,
    response: str,
    context: list,
    llm_client: LLMClient,
) -> float:
    """RAGAS faithfulness for RAG mode. Falls back to DeepEval, then LLM judge."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness as ragas_faithfulness
        from datasets import Dataset
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [context[:3]],
        }
        ds = Dataset.from_dict(data)
        result = ragas_evaluate(ds, metrics=[ragas_faithfulness])
        score = float(result["faithfulness"])
        return round(score, 4)
    except Exception:
        pass

    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        metric = FaithfulnessMetric(threshold=0.5)
        test_case = LLMTestCase(input=query, actual_output=response, retrieval_context=context[:3])
        metric.measure(test_case)
        return round(float(metric.score), 4)
    except Exception:
        pass

    faith_prompt = f"""Rate how faithfully this response is grounded in the context (0.0-1.0).
Context: {' '.join(context[:3])[:600]}
Response: {response[:400]}
Return JSON: {{"faithfulness": <0.0-1.0>}}"""
    try:
        data = await llm_client.complete_json(faith_prompt, temperature=0.1, max_tokens=150, task="judge", retries=2)
        return float(data.get("faithfulness", 0.5))
    except Exception:
        return 0.5
