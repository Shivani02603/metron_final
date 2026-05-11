"""
Stage 4f: RAG evaluation — Answer Relevancy + Answer Correctness (hybrid).

Two metrics only — no context required:

  Answer Relevancy  (DeepEval, LLM-based)
    → Is the generated answer on-topic for the question?
    → Runs on ALL RAG conversations (Stream 1 + Stream 2).

  Answer Correctness  (three-way hybrid: exact match + cosine similarity + LLM judge → max of all three)
    → Does the generated answer match the expected answer?
    → Runs only on Stream 2 conversations (ground truth Q&A pairs).
    → Exact match   : catches identical answers with minor formatting differences (instant, no API).
    → Cosine sim    : catches paraphrases / different phrasing of the same fact (local, no API).
    → LLM judge     : catches numerical equivalents, partial answers, edge cases (API call).
    → Final score = max(exact, cosine, llm) so no correct answer is ever penalised.

All context-dependent metrics (faithfulness, context precision, context recall,
contextual relevancy) have been removed — they require real retrieved chunks from
the endpoint which are not reliably available.
"""

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from core.models import (
    Conversation, MetricResult, Persona, RunConfig, TestClass,
)

_PASS_THRESHOLD = 0.5


# ── Shared metadata helper ─────────────────────────────────────────────────────

def _base_meta(conv: Conversation, persona: Optional[Persona], query: str, response: str) -> Dict[str, Any]:
    return {
        "conversation_id": conv.conversation_id,
        "persona_id":      conv.persona_id,
        "persona_name":    conv.persona_name,
        "intent":          persona.intent.value if persona else "genuine",
        "fishbone":        persona.fishbone_dimensions if persona else {},
        "prompt":          query,
        "response":        response,
        "latency_ms":      conv.total_latency_ms,
        "superset":        "rag",
    }


# ── Cosine similarity via sentence-transformers ────────────────────────────────

_embedding_model = None   # loaded once on first use


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using a lightweight sentence
    embedding model (all-MiniLM-L6-v2, ~80 MB, downloaded once).
    Returns a float in [0.0, 1.0].
    """
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        if _embedding_model is None:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _embedding_model.encode([text_a, text_b], convert_to_tensor=True)
        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[RAG/Cosine] Embedding similarity failed: {e}")
        return 0.0


# ── Exact match ───────────────────────────────────────────────────────────────

def _exact_match(text_a: str, text_b: str) -> float:
    """
    Normalised exact match: lowercase, strip punctuation and extra whitespace.
    Returns 1.0 if texts match after normalisation, 0.0 otherwise.
    """
    import re
    def _normalise(t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)   # strip punctuation
        t = re.sub(r"\s+", " ", t)      # collapse whitespace
        return t

    return 1.0 if _normalise(text_a) == _normalise(text_b) else 0.0


# ── Answer Correctness — three-way hybrid ─────────────────────────────────────

async def _run_answer_correctness(
    conv: Conversation,
    persona: Optional[Persona],
    llm_client: Any,
    query: str,
    answer: str,
    expected: str,
) -> MetricResult:
    """
    Three-way hybrid answer correctness — final score = max(exact, cosine, llm):

    1. Exact Match       — normalised string comparison (fastest, no API)
                           catches: identical answers with minor formatting differences
    2. Cosine Similarity — sentence embedding similarity (local, no API)
                           catches: paraphrases, same fact in different words
    3. LLM Judge         — instructed to accept all valid answer forms (API call)
                           catches: numerical equivalents, partial answers, edge cases

    Taking max means a correct answer is never penalised just because one
    method didn't recognise it.
    """
    base = _base_meta(conv, persona, query, answer)
    loop = asyncio.get_running_loop()

    # ── 1. Exact match (sync, instant) ───────────────────────────────────────
    exact_score = _exact_match(answer, expected)

    # Short-circuit: perfect exact match → no need for cosine or LLM
    if exact_score == 1.0:
        return MetricResult(
            **base,
            metric_name="rag_answer_correctness",
            score=1.0,
            passed=True,
            reason="exact=1.000 — normalised exact match",
        )

    # ── 2. Cosine similarity (sync in executor, no API) ───────────────────────
    cosine_score = await loop.run_in_executor(
        None, _cosine_similarity, answer, expected
    )

    # ── 3. LLM judge ──────────────────────────────────────────────────────────
    judge_prompt = f"""You are evaluating whether a RAG system produced a correct answer.

Question: {query}

Expected Answer: {expected}

Generated Answer: {answer}

Rules for scoring:
- Accept ALL valid phrasings of the same fact as correct (score >= 0.8)
- "6 months" = "six months" = "180 days" = "a six-month period" — all correct
- An answer containing the correct key fact plus extra context is still correct
- An answer more detailed than expected but factually correct is still correct
- Only score low (< 0.5) if the key fact is wrong, contradicts the expected answer, or is completely missing
- Score 0.0 ONLY if the answer is entirely wrong or completely off-topic

Score from 0.0 to 1.0 and return JSON:
{{"score": <float 0.0-1.0>, "reason": "<one sentence: what matched or what was wrong>"}}"""

    llm_score = 0.0
    llm_reason = ""
    try:
        data = await llm_client.complete_json(
            judge_prompt, temperature=0.0, max_tokens=150, task="judge"
        )
        llm_score = float(data.get("score", 0.0))
        llm_score = max(0.0, min(1.0, llm_score))
        llm_reason = data.get("reason", "")
    except Exception as e:
        print(f"[RAG/Correctness] LLM judge failed, falling back to cosine: {e}")
        llm_reason = f"LLM judge unavailable: {str(e)[:80]}"

    # ── 4. Final score = max of all three ────────────────────────────────────
    final_score = max(exact_score, cosine_score, llm_score)
    reason = (
        f"exact={exact_score:.3f}, cosine={cosine_score:.3f}, "
        f"llm={llm_score:.3f}, final=max → {final_score:.3f}. {llm_reason}"
    )

    return MetricResult(
        **base,
        metric_name="rag_answer_correctness",
        score=round(final_score, 4),
        passed=final_score >= _PASS_THRESHOLD,
        reason=reason,
    )


# ── Hallucination — ground truth based ───────────────────────────────────────

async def _run_hallucination(
    conv: Conversation,
    persona: Optional[Persona],
    llm_client: Any,
    query: str,
    answer: str,
    expected: str,
) -> MetricResult:
    """
    LLM judge checks if the response contains information that is not present in
    or contradicts the ground truth answer.
    Score is inverse hallucination — 1.0 means no hallucination, 0.0 means heavy hallucination.
    """
    base = _base_meta(conv, persona, query, answer)

    judge_prompt = f"""You are evaluating whether a RAG system hallucinated in its response.

Question: {query}

Ground Truth Answer: {expected}

Generated Answer: {answer}

Your task: Check if the Generated Answer contains any facts, claims, or information that are NOT present in the Ground Truth Answer or that directly contradict it.

Rules:
- If the Generated Answer only contains information that is supported by or consistent with the Ground Truth, score HIGH (close to 1.0)
- If the Generated Answer adds extra facts or details NOT in the Ground Truth, score LOW (close to 0.0)
- If the Generated Answer contradicts the Ground Truth, score 0.0
- Minor rephrasing or synonyms of the same fact are NOT hallucination
- Additional context that is a reasonable elaboration of the ground truth without adding new unverifiable claims is acceptable

Score from 0.0 to 1.0 (1.0 = no hallucination, 0.0 = severe hallucination) and return JSON:
{{"score": <float 0.0-1.0>, "reason": "<one sentence: what was hallucinated or why it passed>"}}"""

    try:
        data = await llm_client.complete_json(
            judge_prompt, temperature=0.0, max_tokens=150, task="judge"
        )
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = data.get("reason", "")
    except Exception as e:
        return MetricResult(
            **base,
            metric_name="rag_hallucination",
            score=0.0, passed=False, reason="",
            skipped=True,
            skip_reason=f"LLM judge unavailable: {str(e)[:120]}",
        )

    return MetricResult(
        **base,
        metric_name="rag_hallucination",
        score=round(score, 4),
        passed=score >= _PASS_THRESHOLD,
        reason=reason,
    )


# ── Completeness — ground truth based ─────────────────────────────────────────

async def _run_completeness(
    conv: Conversation,
    persona: Optional[Persona],
    llm_client: Any,
    query: str,
    answer: str,
    expected: str,
) -> MetricResult:
    """
    LLM judge checks if the response covers all key facts/points present in the
    ground truth answer. Did the model miss anything important?
    """
    base = _base_meta(conv, persona, query, answer)

    judge_prompt = f"""You are evaluating whether a RAG system's response is complete.

Question: {query}

Ground Truth Answer: {expected}

Generated Answer: {answer}

Your task: Check if the Generated Answer covers all the key facts and points present in the Ground Truth Answer.

Rules:
- If the Generated Answer covers all key facts from the Ground Truth, score HIGH (close to 1.0)
- If the Generated Answer misses some key facts from the Ground Truth, score proportionally lower
- If the Generated Answer misses most or all key facts, score LOW (close to 0.0)
- Minor wording differences are fine — focus on whether the key information is present
- Extra information in the Generated Answer beyond the Ground Truth does not affect the score negatively

Score from 0.0 to 1.0 (1.0 = fully complete, 0.0 = completely missing key facts) and return JSON:
{{"score": <float 0.0-1.0>, "reason": "<one sentence: what key facts were covered or missed>"}}"""

    try:
        data = await llm_client.complete_json(
            judge_prompt, temperature=0.0, max_tokens=150, task="judge"
        )
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = data.get("reason", "")
    except Exception as e:
        return MetricResult(
            **base,
            metric_name="rag_completeness",
            score=0.0, passed=False, reason="",
            skipped=True,
            skip_reason=f"LLM judge unavailable: {str(e)[:120]}",
        )

    return MetricResult(
        **base,
        metric_name="rag_completeness",
        score=round(score, 4),
        passed=score >= _PASS_THRESHOLD,
        reason=reason,
    )


# ── Factual Consistency — ground truth based ──────────────────────────────────

async def _run_factual_consistency(
    conv: Conversation,
    persona: Optional[Persona],
    llm_client: Any,
    query: str,
    answer: str,
    expected: str,
) -> MetricResult:
    """
    LLM judge checks if specific verifiable facts (numbers, dates, names, quantities)
    in the response match the ground truth answer.
    """
    base = _base_meta(conv, persona, query, answer)

    judge_prompt = f"""You are evaluating whether a RAG system's response is factually consistent with the ground truth.

Question: {query}

Ground Truth Answer: {expected}

Generated Answer: {answer}

Your task: Focus specifically on verifiable facts — numbers, dates, names, quantities, percentages, durations, and proper nouns. Check if these match between the Generated Answer and Ground Truth Answer.

Rules:
- If all specific facts in the Generated Answer match the Ground Truth, score HIGH (close to 1.0)
- If some facts are wrong or inconsistent, score proportionally lower
- If the Generated Answer has no specific facts to check, score 1.0 (nothing to be wrong)
- "6 months" and "180 days" and "six months" are all consistent — equivalent forms are acceptable
- Only penalise when a fact clearly contradicts the Ground Truth (e.g., wrong number, wrong name)

Score from 0.0 to 1.0 (1.0 = all facts consistent, 0.0 = facts are wrong) and return JSON:
{{"score": <float 0.0-1.0>, "reason": "<one sentence: which facts matched or which were inconsistent>"}}"""

    try:
        data = await llm_client.complete_json(
            judge_prompt, temperature=0.0, max_tokens=150, task="judge"
        )
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = data.get("reason", "")
    except Exception as e:
        return MetricResult(
            **base,
            metric_name="rag_factual_consistency",
            score=0.0, passed=False, reason="",
            skipped=True,
            skip_reason=f"LLM judge unavailable: {str(e)[:120]}",
        )

    return MetricResult(
        **base,
        metric_name="rag_factual_consistency",
        score=round(score, 4),
        passed=score >= _PASS_THRESHOLD,
        reason=reason,
    )


# ── Answer Relevancy — DeepEval (question + answer only) ──────────────────────

async def _run_answer_relevancy(
    conv: Conversation,
    persona: Optional[Persona],
    deval_model: Any,
    query: str,
    answer: str,
) -> MetricResult:
    """
    DeepEval AnswerRelevancyMetric: is the answer on-topic for the question?
    Needs only question + answer — no context required.
    """
    from deepeval.metrics import AnswerRelevancyMetric  # type: ignore
    from deepeval.test_case import LLMTestCase          # type: ignore

    base = _base_meta(conv, persona, query, answer)
    loop = asyncio.get_running_loop()

    try:
        test_case = LLMTestCase(input=query, actual_output=answer)
        metric    = AnswerRelevancyMetric(model=deval_model, threshold=_PASS_THRESHOLD)
        await asyncio.wait_for(
            loop.run_in_executor(None, metric.measure, test_case),
            timeout=120,
        )
        score = float(metric.score or 0.0)
        return MetricResult(
            **base,
            metric_name="rag_answer_relevancy",
            score=round(score, 4),
            passed=metric.is_successful(),
            reason=metric.reason or "",
        )
    except Exception as e:
        return MetricResult(
            **base,
            metric_name="rag_answer_relevancy",
            score=0.0, passed=False, reason="",
            skipped=True,
            skip_reason=f"DeepEval error: {str(e)[:120]}",
        )


# ── Main entry point ───────────────────────────────────────────────────────────

async def evaluate_rag(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: Any,
) -> List[MetricResult]:
    """
    RAG evaluation — two metrics, no context dependency:

      rag_answer_relevancy   → all RAG functional conversations (Stream 1 + Stream 2)
      rag_answer_correctness → Stream 2 only (conversations with expected_answer)

    Returns list of MetricResult objects tagged superset="rag".
    """
    from core.deepeval_azure import make_deepeval_model  # type: ignore
    from stages.s4_evaluation.functional import _set_azure_env

    _set_azure_env(config)

    # All functional conversations from both streams
    rag_convs = [
        c for c in conversations
        if c.test_class == TestClass.FUNCTIONAL and c.turns
    ]

    if not rag_convs:
        print("[RAG Eval] No functional conversations — skipping RAG metrics.")
        return []

    persona_map = {p.persona_id: p for p in personas}
    all_results: List[MetricResult] = []

    # ── Answer Relevancy — all conversations ──────────────────────────────────
    try:
        deval_model = make_deepeval_model(
            provider_name=config.llm_provider,
            api_key=config.llm_api_key,
            azure_endpoint=getattr(config, "azure_endpoint", "") or "",
        )
    except Exception as e:
        print(f"[RAG/DeepEval] Could not init model: {e}")
        deval_model = None

    if deval_model:
        sem = asyncio.Semaphore(3)

        async def _bounded_relevancy(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem:
                return await _run_answer_relevancy(
                    conv, persona, deval_model, t.query, t.response
                )

        relevancy_results = await asyncio.gather(*[_bounded_relevancy(c) for c in rag_convs])
        all_results.extend(relevancy_results)
        passed_rel = sum(1 for r in relevancy_results if r.passed)
        print(f"[RAG Eval] Answer relevancy: {passed_rel}/{len(relevancy_results)} passed")

    # ── Answer Correctness — Stream 2 only (has expected_answer) ─────────────
    correctness_convs = [
        c for c in rag_convs
        if c.turns[0].expected_answer
    ]

    if correctness_convs:
        # Run all four metric types IN PARALLEL — they are independent of each other.
        # Previously sequential: correctness → hallucination → completeness → factual
        # meant each gather had to fully complete before the next started (~4x slower).
        sem2 = asyncio.Semaphore(3)

        async def _bounded_correctness(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem2:
                return await _run_answer_correctness(
                    conv, persona, llm_client,
                    t.query, t.response, t.expected_answer,
                )

        async def _bounded_hallucination(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem2:
                return await _run_hallucination(
                    conv, persona, llm_client,
                    t.query, t.response, t.expected_answer,
                )

        async def _bounded_completeness(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem2:
                return await _run_completeness(
                    conv, persona, llm_client,
                    t.query, t.response, t.expected_answer,
                )

        async def _bounded_factual(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem2:
                return await _run_factual_consistency(
                    conv, persona, llm_client,
                    t.query, t.response, t.expected_answer,
                )

        (
            correctness_results,
            hallucination_results,
            completeness_results,
            factual_results,
        ) = await asyncio.gather(
            asyncio.gather(*[_bounded_correctness(c)  for c in correctness_convs]),
            asyncio.gather(*[_bounded_hallucination(c) for c in correctness_convs]),
            asyncio.gather(*[_bounded_completeness(c)  for c in correctness_convs]),
            asyncio.gather(*[_bounded_factual(c)       for c in correctness_convs]),
        )

        all_results.extend(correctness_results)
        all_results.extend(hallucination_results)
        all_results.extend(completeness_results)
        all_results.extend(factual_results)

        passed_cor = sum(1 for r in correctness_results if r.passed)
        passed_hal = sum(1 for r in hallucination_results if r.passed)
        passed_com = sum(1 for r in completeness_results if r.passed)
        passed_fac = sum(1 for r in factual_results if r.passed)
        print(f"[RAG Eval] Answer correctness: {passed_cor}/{len(correctness_results)} passed")
        print(f"[RAG Eval] Hallucination: {passed_hal}/{len(hallucination_results)} passed")
        print(f"[RAG Eval] Completeness: {passed_com}/{len(completeness_results)} passed")
        print(f"[RAG Eval] Factual consistency: {passed_fac}/{len(factual_results)} passed")

    gt_count      = sum(1 for c in rag_convs if c.persona_id == "ground_truth_stream")
    persona_count = len(rag_convs) - gt_count
    print(f"[RAG Eval] Complete — {len(all_results)} metric results "
          f"({persona_count} persona-based, {gt_count} ground-truth)")
    return all_results
