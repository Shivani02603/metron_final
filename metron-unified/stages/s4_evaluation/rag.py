"""
Stage 4f: RAG evaluation using RAGAS + DeepEval.

Tool split:
  RAGAS    → faithfulness, context_recall, context_precision   (batch, LLM-based)
  DeepEval → answer_relevancy, contextual_relevancy            (per-conv, LLM-based)

Both tools route through Azure GPT-4o.
Only runs on functional conversations where retrieved_context is present.
context_recall and context_precision require ground truth (expected_answer on turn 1).

RAGAS 0.4.x API: uses EvaluationDataset + SingleTurnSample.
Falls back to HuggingFace Dataset (0.1.x) if new API unavailable.
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
    """Build a RAGAS-compatible LLM wrapper using Azure GPT-4o."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import AzureChatOpenAI

    azure_llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
        max_tokens=1024,
    )
    return LangchainLLMWrapper(azure_llm)


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
        # Fallback: ragas 0.1.x HuggingFace Dataset API
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
        # Normalize column names: 0.4.x may use different names
        rename_map = {
            # 0.4.x names → canonical names
            "user_input":        "question",
            "response":          "answer",
            "retrieved_contexts":"contexts",
            "reference":         "ground_truth",
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
) -> Tuple[Optional[Any], bool, str]:
    """
    Run RAGAS evaluate() synchronously.
    Returns (result_df, has_ground_truth, status_msg).
    """
    try:
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import faithfulness  # type: ignore

        ragas_llm = _build_azure_ragas_llm()

        has_ground_truth = any(gt.strip() for gt in ground_truths)

        # Configure faithfulness metric with our Azure LLM
        # In 0.4.x, metric instances may need to be constructed fresh with llm param,
        # or the .llm setter still works — we try setter first, then constructor kwarg.
        active_metrics = []
        try:
            faithfulness.llm = ragas_llm
            active_metrics.append(faithfulness)
        except AttributeError:
            from ragas.metrics import Faithfulness  # type: ignore
            active_metrics.append(Faithfulness(llm=ragas_llm))

        if has_ground_truth:
            try:
                from ragas.metrics import context_recall, context_precision  # type: ignore
                context_recall.llm    = ragas_llm
                context_precision.llm = ragas_llm
                active_metrics += [context_recall, context_precision]
            except (ImportError, AttributeError):
                try:
                    from ragas.metrics import ContextRecall, ContextPrecision  # type: ignore
                    active_metrics += [
                        ContextRecall(llm=ragas_llm),
                        ContextPrecision(llm=ragas_llm),
                    ]
                except Exception:
                    pass  # No recall/precision available — run faithfulness only

        dataset, column_style = _build_ragas_dataset(questions, answers, contexts, ground_truths)

        result = ragas_evaluate(dataset, metrics=active_metrics)
        df     = _extract_ragas_scores(result, column_style)
        return df, has_ground_truth, "ok"

    except KeyError as e:
        msg = f"Azure env var missing: {e}"
        print(f"[RAG/RAGAS] {msg}")
        return None, False, msg
    except Exception as e:
        msg = str(e)[:200]
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
    """
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric  # type: ignore
    from deepeval.test_case import LLMTestCase  # type: ignore

    base    = _base_meta(conv, persona, query, answer)
    results = []
    loop    = asyncio.get_event_loop()

    # Ensure context is a non-empty list (DeepEval requires at least one string)
    safe_context = context if context else ["(no context retrieved)"]

    for metric_cls, metric_name in [
        (AnswerRelevancyMetric,    "rag_answer_relevancy"),
        (ContextualRelevancyMetric,"rag_context_relevancy"),
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
            results.append(MetricResult(
                **base,
                metric_name=metric_name,
                score=0.0,
                passed=False,
                reason=f"Error: {str(e)[:120]}",
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
    Returns list of MetricResult objects.
    """
    from core.deepeval_azure import make_deepeval_azure_model  # type: ignore
    from stages.s4_evaluation.functional import _set_azure_env

    _set_azure_env(config)

    # Only evaluate functional conversations that have retrieved_context
    rag_convs = [
        c for c in conversations
        if c.test_class == TestClass.FUNCTIONAL
        and c.turns
        and c.turns[0].retrieved_context
    ]

    if not rag_convs:
        print("[RAG Eval] No conversations with retrieved_context — skipping RAG metrics.")
        return []

    persona_map = {p.persona_id: p for p in personas}

    # Build parallel lists for RAGAS batch evaluation
    questions:     List[str]       = []
    answers:       List[str]       = []
    contexts:      List[List[str]] = []
    ground_truths: List[str]       = []

    for conv in rag_convs:
        t = conv.turns[0]
        questions.append(t.query)
        answers.append(t.response)
        contexts.append(t.retrieved_context or [])
        ground_truths.append(t.expected_answer or "")

    all_results: List[MetricResult] = []

    # ── RAGAS batch evaluation ─────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    df, has_gt, ragas_status = await loop.run_in_executor(
        None, _run_ragas, questions, answers, contexts, ground_truths
    )

    if df is not None:
        ragas_metric_cols = {
            "faithfulness":      "rag_faithfulness",
            "context_recall":    "rag_context_recall",
            "context_precision": "rag_context_precision",
        }
        for idx, conv in enumerate(rag_convs):
            persona = persona_map.get(conv.persona_id)
            t       = conv.turns[0]
            base    = _base_meta(conv, persona, t.query, t.response)
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            for col, metric_name in ragas_metric_cols.items():
                if col not in df.columns:
                    continue
                raw_score = row[col]
                # Guard against NaN
                import math
                score = float(raw_score) if (raw_score == raw_score and not (isinstance(raw_score, float) and math.isnan(raw_score))) else 0.0
                all_results.append(MetricResult(
                    **base,
                    metric_name=metric_name,
                    score=round(score, 4),
                    passed=score >= 0.5,
                    reason=f"RAGAS {col}: {score:.3f}",
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
