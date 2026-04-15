"""
Stage 4f: RAG evaluation using RAGAS + DeepEval.

Tool split:
  RAGAS    → non-LLM metrics first (no API calls required):
               NonLLMContextPrecisionWithReference  (RAGAS 0.2.x+)
               NonLLMContextRecall                  (RAGAS 0.2.x+)
             LLM metric added only when Azure env is confirmed available:
               Faithfulness (LLM-based, Azure GPT-4o)
  DeepEval → answer_relevancy, contextual_relevancy (per-conv, Azure GPT-4o)

Fix 18: only the RAGAS metrics listed in config.ragas_metrics are run.
Fix 16: empty context uses "NO_CONTEXT_RETRIEVED" placeholder with an explicit
        warning instead of the silent [""] fallback.
Fix 20: asyncio.get_running_loop() replaces deprecated get_event_loop().
Fix 36: DeepEval exceptions → skipped MetricResult instead of silent drop.

Non-LLM metrics always run regardless of Azure configuration.
LLM metrics are additive — failure does not suppress non-LLM results.
"""

from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from core.models import (
    Conversation, MetricResult, Persona, RunConfig, TestClass,
)


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


def _build_azure_ragas_llm():
    """Build a RAGAS-compatible LLM wrapper using the shared Azure LangChain client."""
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from stages.s4_evaluation.functional import _build_azure_langchain_llm
    return LangchainLLMWrapper(_build_azure_langchain_llm())


def _build_ragas_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
):
    """
    Build RAGAS dataset — tries 0.4.x EvaluationDataset first, falls back to HF Dataset.
    Returns (dataset, column_style) where column_style is "new" or "old".
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample  # type: ignore
        samples = []
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
            sample = SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=ctx if ctx else [""],
                reference=gt if gt and gt.strip() else None,
            )
            samples.append(sample)
        return EvaluationDataset(samples=samples), "new"
    except (ImportError, AttributeError):
        from datasets import Dataset  # type: ignore
        data = {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(data), "old"


def _extract_ragas_scores(result: Any, column_style: str) -> Optional[Any]:
    """
    Extract a pandas DataFrame from the RAGAS result object.
    Handles both 0.4.x and 0.1.x result formats.
    """
    try:
        df = result.to_pandas()
        rename_map = {
            "user_input":         "question",
            "response":           "answer",
            "retrieved_contexts": "contexts",
            "reference":          "ground_truth",
        }
        df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        print(f"[RAG/RAGAS] Could not convert result to DataFrame: {e}")
        return None


def _run_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    active_ragas_metrics: set,
) -> Tuple[Optional[Any], bool, str]:
    """
    Run RAGAS evaluate() synchronously.

    Fix 18: active_ragas_metrics controls which metrics are attempted.

    Metric priority:
      1. Non-LLM metrics (no Azure required, always attempted when in active set):
           NonLLMContextPrecisionWithReference  — requires ground_truth
           NonLLMContextRecall                  — requires ground_truth
      2. LLM metric (Azure required, added only when in active set and env set):
           Faithfulness

    Returns (result_df, has_ground_truth, status_msg).
    """
    try:
        from ragas import evaluate as ragas_evaluate  # type: ignore

        has_ground_truth = any(gt.strip() for gt in ground_truths)
        active_metrics: List[Any] = []

        # ── 1. Non-LLM metrics ────────────────────────────────────────────────
        if has_ground_truth and "context_precision" in active_ragas_metrics:
            try:
                from ragas.metrics import NonLLMContextPrecisionWithReference  # type: ignore
                active_metrics.append(NonLLMContextPrecisionWithReference())
                print("[RAG/RAGAS] Added NonLLMContextPrecisionWithReference")
            except (ImportError, Exception) as e:
                print(f"[RAG/RAGAS] NonLLMContextPrecisionWithReference unavailable: {e}")

        if has_ground_truth and "context_recall" in active_ragas_metrics:
            try:
                from ragas.metrics import NonLLMContextRecall  # type: ignore
                active_metrics.append(NonLLMContextRecall())
                print("[RAG/RAGAS] Added NonLLMContextRecall")
            except (ImportError, Exception) as e:
                print(f"[RAG/RAGAS] NonLLMContextRecall unavailable: {e}")

        # ── 2. LLM metric: faithfulness ───────────────────────────────────────
        azure_ready = bool(
            os.environ.get("AZURE_OPENAI_ENDPOINT") and
            os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if "faithfulness" in active_ragas_metrics and azure_ready:
            try:
                ragas_llm = _build_azure_ragas_llm()
                try:
                    from ragas.metrics import faithfulness  # type: ignore
                    faithfulness.llm = ragas_llm
                    active_metrics.append(faithfulness)
                except AttributeError:
                    from ragas.metrics import Faithfulness  # type: ignore
                    active_metrics.append(Faithfulness(llm=ragas_llm))
                print("[RAG/RAGAS] Added Faithfulness (LLM-based, Azure)")
            except Exception as e:
                print(f"[RAG/RAGAS] Faithfulness metric setup failed (non-LLM metrics still run): {e}")
        elif "faithfulness" in active_ragas_metrics:
            print("[RAG/RAGAS] Azure env not set — skipping Faithfulness (non-LLM metrics still run)")

        if not active_metrics:
            return None, has_ground_truth, "No RAGAS metrics could be initialised (ground_truth required for non-LLM metrics)"

        dataset, column_style = _build_ragas_dataset(questions, answers, contexts, ground_truths)
        result = ragas_evaluate(dataset, metrics=active_metrics)
        df = _extract_ragas_scores(result, column_style)
        return df, has_ground_truth, "ok"

    except Exception as e:
        msg = str(e)[:300]
        print(f"[RAG/RAGAS] Evaluation failed: {msg}")
        return None, False, msg


async def _run_deepeval_metrics(
    conv: Conversation,
    persona: Optional[Persona],
    deval_model: Any,
    query: str,
    answer: str,
    context: List[str],
    expected: str,
) -> List[MetricResult]:
    """
    Run DeepEval RAG metrics per conversation:
      - AnswerRelevancyMetric
      - ContextualRelevancyMetric

    Fix 36: exceptions produce skipped MetricResult instead of score=0.0.
    """
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric  # type: ignore
    from deepeval.test_case import LLMTestCase  # type: ignore

    base    = _base_meta(conv, persona, query, answer)
    results = []
    loop    = asyncio.get_running_loop()   # Fix 20

    safe_context = context if context else ["(no context retrieved)"]

    for metric_cls, metric_name in [
        (AnswerRelevancyMetric,     "rag_answer_relevancy"),
        (ContextualRelevancyMetric, "rag_context_relevancy"),
    ]:
        try:
            tc_kwargs: Dict[str, Any] = {
                "input":             query,
                "actual_output":     answer,
                "retrieval_context": safe_context,
            }
            if expected:
                tc_kwargs["expected_output"] = expected

            test_case = LLMTestCase(**tc_kwargs)
            metric    = metric_cls(model=deval_model, threshold=0.5)
            await loop.run_in_executor(None, metric.measure, test_case)
            score = float(metric.score or 0.0)
            results.append(MetricResult(
                **base,
                metric_name=metric_name,
                score=round(score, 4),
                passed=metric.is_successful(),
                reason=metric.reason or "",
            ))
        except Exception as e:
            # Fix 36: skipped result instead of silent failure
            results.append(MetricResult(
                **base,
                metric_name=metric_name,
                score=0.0, passed=False, reason="",
                skipped=True,
                skip_reason=f"DeepEval error: {str(e)[:120]}",
            ))

    return results


async def evaluate_rag(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: Any,
) -> List[MetricResult]:
    """
    Run RAG evaluation using RAGAS (batch) + DeepEval (per-conv).

    Fix 18: only metrics listed in config.ragas_metrics are run.
    Fix 16: missing context logs a warning and uses "NO_CONTEXT_RETRIEVED" placeholder.
    Returns list of MetricResult objects.
    """
    from core.deepeval_azure import make_deepeval_azure_model  # type: ignore
    from stages.s4_evaluation.functional import _set_azure_env

    _set_azure_env(config)

    # Fix 18: build active metric set from config (default to all if not specified)
    _default_ragas = {"faithfulness", "answer_relevancy", "context_recall", "context_precision"}
    active_ragas = set(getattr(config, "ragas_metrics", None) or _default_ragas)

    rag_convs = [
        c for c in conversations
        if c.test_class == TestClass.FUNCTIONAL
        and c.turns
        and (c.turns[0].retrieved_context or c.turns[0].expected_answer)
    ]

    if not rag_convs:
        print("[RAG Eval] No functional conversations with context or expected_answer — skipping RAG metrics.")
        return []

    persona_map = {p.persona_id: p for p in personas}

    questions:     List[str]       = []
    answers:       List[str]       = []
    contexts:      List[List[str]] = []
    ground_truths: List[str]       = []

    for conv in rag_convs:
        t = conv.turns[0]
        questions.append(t.query)
        answers.append(t.response)

        # Fix 16: warn when context is missing; use explicit placeholder
        if t.retrieved_context:
            contexts.append(t.retrieved_context)
        else:
            print(
                f"[RAG Eval] Warning: no retrieved_context for conv {conv.conversation_id[:8]} "
                f"(persona: {conv.persona_name}). RAGAS will use NO_CONTEXT_RETRIEVED placeholder — "
                f"context precision/recall scores will reflect a context miss, not a real retrieval."
            )
            contexts.append(["NO_CONTEXT_RETRIEVED"])

        ground_truths.append(t.expected_answer or "")

    all_results: List[MetricResult] = []

    # ── RAGAS batch evaluation ─────────────────────────────────────────────────
    loop = asyncio.get_running_loop()   # Fix 20
    df, has_gt, ragas_status = await loop.run_in_executor(
        None, _run_ragas, questions, answers, contexts, ground_truths, active_ragas
    )

    if df is not None:
        ragas_metric_cols = {
            "faithfulness":      "rag_faithfulness",
            "context_recall":    "rag_context_recall",
            "context_precision": "rag_context_precision",
            "non_llm_context_precision_with_reference": "rag_context_precision",
            "non_llm_context_recall":                   "rag_context_recall",
        }
        for idx, conv in enumerate(rag_convs):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            base    = _base_meta(conv, persona, t.query, t.response)
            if idx >= len(df):
                continue
            row = df.iloc[idx]

            # Flag context miss in reason
            missing_ctx = contexts[idx] == ["NO_CONTEXT_RETRIEVED"]

            for col, metric_name in ragas_metric_cols.items():
                if col not in df.columns:
                    continue
                raw_score = row[col]
                import math
                score = float(raw_score) if (raw_score == raw_score and not (isinstance(raw_score, float) and math.isnan(raw_score))) else 0.0
                reason = f"RAGAS {col}: {score:.3f}"
                if missing_ctx:
                    reason += " (Warning: no context retrieved from endpoint)"
                all_results.append(MetricResult(
                    **base,
                    metric_name=metric_name,
                    score=round(score, 4),
                    passed=score >= 0.5,
                    reason=reason,
                ))
    else:
        print(f"[RAG/RAGAS] Skipped — {ragas_status}")

    # ── DeepEval per-conversation ─────────────────────────────────────────────
    try:
        deval_model = make_deepeval_azure_model()
    except Exception as e:
        print(f"[RAG/DeepEval] Could not init model: {e}")
        deval_model = None

    if deval_model:
        sem = asyncio.Semaphore(3)

        async def _bounded_deepeval(conv: Conversation):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            async with sem:
                return await _run_deepeval_metrics(
                    conv, persona, deval_model,
                    t.query, t.response,
                    t.retrieved_context or [],
                    t.expected_answer or "",
                )

        batches = await asyncio.gather(*[_bounded_deepeval(c) for c in rag_convs])
        for batch in batches:
            all_results.extend(batch)

    print(f"[RAG Eval] Complete — {len(all_results)} metric results "
          f"({len(rag_convs)} conversations, ground_truth={'yes' if has_gt else 'no'})")
    return all_results
