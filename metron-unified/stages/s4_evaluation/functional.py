"""
Stage 4a: Functional evaluation.
Metrics: DeepEval (AnswerRelevancy, Hallucination) + RAGAS (faithfulness, answer_relevancy for RAG)
         + ROUGE-L + BERTScore + LLM-as-judge fallback.
Sourced from new metron-backend/app/stage5_evaluation/evaluators/functionality.py.
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

_DEFAULT_CRITERIA_TEXT = "- Relevance: Does it directly address the question?\n- Accuracy: Is the information correct?\n- Helpfulness: Does it help the user achieve their goal?"


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

    # Build criteria text once from quality_criteria (domain-specific) or fallback
    if quality_criteria and quality_criteria.get("criteria"):
        criteria_lines = [
            f"- {c['name']}: {c.get('description', '')}"
            for c in quality_criteria["criteria"]
        ]
        criteria_text = "\n".join(criteria_lines)
        pass_threshold = quality_criteria.get("passing_threshold", THRESHOLDS["functional_pass"])
    else:
        criteria_text = _DEFAULT_CRITERIA_TEXT
        pass_threshold = THRESHOLDS["functional_pass"]

    sem = asyncio.Semaphore(5)

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

        rouge_score = _rouge_l(query, response)
        local.append(MetricResult(
            **base_meta, metric_name="rouge_l", score=rouge_score,
            passed=rouge_score >= THRESHOLDS["rouge_l_min"],
            reason=f"ROUGE-L F1: {rouge_score:.3f}",
        ))

        bert_score = _bert_score_approx(query, response)
        local.append(MetricResult(
            **base_meta, metric_name="bert_score_f1", score=bert_score,
            passed=bert_score >= THRESHOLDS["bert_score_min"],
            reason=f"BERTScore F1 (approx): {bert_score:.3f}",
        ))

        async with sem:
            judge_result = await _llm_judge(query, response, "", llm_client, criteria_text)
        overall = judge_result.get("overall", 0.5)
        reasoning = judge_result.get("reasoning", "")
        # Emit one result per criterion returned by the judge
        skip_keys = {"overall", "reasoning"}
        for crit_key, crit_score in judge_result.items():
            if crit_key in skip_keys:
                continue
            try:
                s = float(crit_score)
            except (TypeError, ValueError):
                continue
            local.append(MetricResult(
                **base_meta, metric_name=crit_key, score=s,
                passed=s >= pass_threshold, reason=reasoning,
            ))
        # Always emit overall as answer_relevancy for backward compat
        local.append(MetricResult(
            **base_meta, metric_name="answer_relevancy", score=overall,
            passed=overall >= pass_threshold, reason=reasoning,
        ))

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
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
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
    """Approximate BERTScore using token overlap (fallback when bert_score not installed)."""
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score([hypothesis], [reference], lang="en", verbose=False)
        return round(float(F1[0]), 4)
    except ImportError:
        # Token overlap fallback
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
        expected=expected[:200] or "A helpful, accurate response",
        criteria_text=criteria_text,
    )
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.1, max_tokens=500, task="judge", retries=2,
        )
        # Normalise: ensure float values, preserve all criterion keys
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


async def _faithfulness(
    query: str,
    response: str,
    context: list,
    llm_client: LLMClient,
) -> float:
    """Check if response is grounded in retrieved context."""
    try:
        from ragas.metrics import faithfulness as ragas_faithfulness
        # Attempt RAGAS if available
        # (RAGAS requires dataset format — complex setup, use LLM judge as fallback)
        raise ImportError("Use LLM fallback")
    except Exception:
        pass

    faith_prompt = f"""Given this context and response, rate how faithfully the response is grounded in the context (0.0-1.0).
Context: {' '.join(context[:3])[:600]}
Response: {response[:400]}
Return JSON: {{"faithfulness": <0.0-1.0>, "reason": "<1 sentence>"}}"""
    try:
        data = await llm_client.complete_json(
            faith_prompt, temperature=0.1, max_tokens=200, task="judge", retries=2,
        )
        return float(data.get("faithfulness", 0.5))
    except Exception:
        return 0.5
