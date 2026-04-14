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
    Run RAGAS metrics for one conversation.

    Metric priority (mirrors rag.py strategy):
      1. Non-LLM metrics first (no Azure required):
           NonLLMContextPrecisionWithReference  — needs ground_truth
           NonLLMContextRecall                  — needs ground_truth
      2. LLM metric (Azure required, added when env vars are present):
           Faithfulness
    Returns dict of metric_name → (score, method).
    Raises on complete failure — caller handles.
    """
    import os, math
    from ragas import evaluate as ragas_evaluate  # type: ignore

    ctx    = context[:3] or [""]
    has_gt = bool(ground_truth and ground_truth.strip())
    metrics_to_run: list = []

    # ── 1. Non-LLM metrics (always attempted when ground truth is available) ──
    if has_gt:
        try:
            from ragas.metrics import NonLLMContextPrecisionWithReference  # type: ignore
            metrics_to_run.append(NonLLMContextPrecisionWithReference())
        except (ImportError, Exception):
            pass

        try:
            from ragas.metrics import NonLLMContextRecall  # type: ignore
            metrics_to_run.append(NonLLMContextRecall())
        except (ImportError, Exception):
            pass

    # ── 2. LLM metric: faithfulness (only when Azure is configured) ───────────
    azure_ready = bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT") and
        os.environ.get("AZURE_OPENAI_API_KEY")
    )
    if azure_ready:
        try:
            from ragas.llms import LangchainLLMWrapper  # type: ignore
            from stages.s4_evaluation.functional import _build_azure_langchain_llm
            ragas_llm = LangchainLLMWrapper(_build_azure_langchain_llm())
            try:
                from ragas.metrics import faithfulness as r_faith  # type: ignore
                r_faith.llm = ragas_llm
                metrics_to_run.append(r_faith)
            except AttributeError:
                from ragas.metrics import Faithfulness  # type: ignore
                metrics_to_run.append(Faithfulness(llm=ragas_llm))
        except Exception as e:
            print(f"[RAGAS/quality] Faithfulness setup failed: {e}")

    if not metrics_to_run:
        raise RuntimeError("No RAGAS metrics available (ground_truth required for non-LLM metrics)")

    # Build dataset
    try:
        from ragas import EvaluationDataset, SingleTurnSample  # type: ignore
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=ctx,
            reference=ground_truth if has_gt else None,
        )
        ds = EvaluationDataset(samples=[sample])
    except (ImportError, AttributeError):
        from datasets import Dataset  # type: ignore
        data: dict = {"question": [query], "answer": [response], "contexts": [ctx]}
        if has_gt:
            data["ground_truth"] = [ground_truth]
        ds = Dataset.from_dict(data)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: ragas_evaluate(ds, metrics=metrics_to_run))

    # Column map covers both LLM-based and non-LLM column names
    col_map = {
        "faithfulness":                             "ragas_faithfulness",
        "context_recall":                           "ragas_context_recall",
        "context_precision":                        "ragas_context_precision",
        "non_llm_context_precision_with_reference": "ragas_context_precision",
        "non_llm_context_recall":                   "ragas_context_recall",
    }
    out: dict[str, tuple[float, str]] = {}

    try:
        df  = result.to_pandas()
        row = df.iloc[0]
        for col, metric_name in col_map.items():
            if col in df.columns and metric_name not in out:
                raw   = row[col]
                score = float(raw) if (raw == raw and not (isinstance(raw, float) and math.isnan(raw))) else 0.0
                out[metric_name] = (round(score, 4), "ragas")
    except Exception:
        # Fallback: dict-style access (ragas 0.1.x)
        for col, metric_name in col_map.items():
            if metric_name in out:
                continue
            try:
                val = result[col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    out[metric_name] = (round(float(val), 4), "ragas")
            except (KeyError, TypeError):
                pass

    return out


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_quality(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """Evaluate quality using DeepEval GEval + RAGAS (tool-only, no LLM fallback)."""
    from stages.s4_evaluation.functional import _set_azure_env
    _set_azure_env(config)

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

        # Skip quality GEval for security conversations — attack prompts (jailbreak /
        # harmful content from AdvBench/HarmBench) trigger Azure content filter when
        # passed as GEval input. Quality metrics (coherence, tone, etc.) are also not
        # meaningful for adversarial test conversations.
        if conv.test_class == TestClass.SECURITY:
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
        # Respect config.use_geval — the UI toggle that disables GEval entirely.
        if criteria_list and getattr(config, "use_geval", True):
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
