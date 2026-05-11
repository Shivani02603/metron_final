"""
Stage 4a: Functional evaluation.
Strict tool-only — no heuristics or hardcoded fallback scores.

Metrics:
  - Hallucination    → DeepEval HallucinationMetric  (Azure OpenAI)
                       Only runs when "hallucination" in config.deepeval_metrics
                       Reference: retrieved_context (RAG) or expected_behavior (non-RAG)
                       Skipped entirely when no reference is available.
  - Answer Relevancy → DeepEval AnswerRelevancyMetric (Azure OpenAI)
                       Only runs when "answer_relevancy" in config.deepeval_metrics
  - Usefulness       → DeepEval GEval (fixed usefulness criterion — NOT domain criteria)
  - LLM Judge        → basic correctness (relevance, accuracy, helpfulness)
                       Runs only when config.enable_judge=True (default)
                       Uses ONLY default criteria — domain GEval criteria live in quality.py
  - Completeness     → LLM-based comparison against expected_behavior (non-RAG only)
  - Consistency      → Cross-turn consistency check for multi-turn conversations

  Skipped metrics: on any tool failure, a MetricResult with skipped=True is recorded
  so the aggregator denominator stays honest.

Error responses (is_error_response=True) skip all metric scoring.
"""

from __future__ import annotations
import asyncio
import re
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    Conversation, MetricResult, Persona, RunConfig,
)
from core.config import THRESHOLDS
from core.deepeval_azure import make_deepeval_model

# ── LLM Judge prompts ─────────────────────────────────────────────────────────

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

COMPLETENESS_PROMPT = """
Compare this AI response to the expected behavior.

QUESTION: {question}
AI RESPONSE: {response}
EXPECTED BEHAVIOR: {expected}

Rate on a scale of 0.0 to 1.0:
- completeness: Does the response cover all key points from the expected behavior?
- answer_similarity: How similar are the response and expected behavior semantically?

Return JSON:
{{
  "completeness": <0.0-1.0>,
  "answer_similarity": <0.0-1.0>,
  "reasoning": "<1-2 sentences>"
}}
"""

CROSS_TURN_CONSISTENCY_PROMPT = """
Review this multi-turn conversation and check for self-contradiction.

{turns_text}

Does the AI contradict itself between turns? Does its tone or persona shift in a way
that suggests it was manipulated?

Return JSON:
{{
  "consistency": <1.0 = fully consistent, 0.0 = direct contradiction>,
  "reasoning": "<1-2 sentences>"
}}
"""

_DEFAULT_CRITERIA_TEXT = (
    "- relevance: Does it directly address the question?\n"
    "- accuracy: Is the information correct?\n"
    "- helpfulness: Does it help the user achieve their goal?"
)


# ── Expected-behavior reference quality check ─────────────────────────────────

# Keywords that signal a behavioral/prescriptive description rather than a factual document.
# If 3+ of these appear in expected_behavior it is directive text, not a reference corpus.
_BEHAVIORAL_DIRECTIVE_KEYWORDS = frozenset({
    "should", "must", "expected to", "required to", "needs to",
    "ought to", "is designed to", "the agent", "the bot", "the assistant",
    "respond with", "respond by", "is supposed to",
})


def _is_factual_reference(text: str) -> bool:
    """
    Return True only when expected_behavior is a factual reference document
    suitable as a hallucination ground-truth (contains specific verifiable
    information), not a behavioral description that says what the agent ought to do.

    Behavioral descriptions ("The assistant should respond professionally…")
    cause false hallucination failures when used as DeepEval reference context
    because any on-topic answer will diverge from the prescriptive wording.
    """
    if not text or len(text) < 100:
        return False
    text_lower = text.lower()
    directive_hits = sum(1 for kw in _BEHAVIORAL_DIRECTIVE_KEYWORDS if kw in text_lower)
    # 2+ directive keywords → predominantly a behavioral description → not a reference.
    # Lowered from 3 to 2: a single "should" + "the assistant" is already enough to
    # identify prescriptive text; using it as a hallucination reference produces false
    # failures because any valid response will diverge from the prescriptive wording.
    if directive_hits >= 2:
        return False
    # Must still be long enough to contain meaningful factual content
    return len(text) >= 200


def _configure_deepeval(llm_provider: str, llm_api_key: str, azure_endpoint: str = "") -> None:
    """Ensure Azure env vars are set so the AzureOpenAI client resolves correctly."""
    try:
        import os
        p = llm_provider.lower()
        if "azure" in p and llm_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = llm_api_key
        if "azure" in p and azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        if "openai" in p and "azure" not in p and llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
    except Exception:
        pass


def _set_azure_env(config: RunConfig) -> None:
    """
    Ensure Azure OpenAI env vars are populated for RAGAS and DeepEval tool calls.
    Reads from RunConfig.llm_api_key / azure_endpoint when provider is Azure.
    Also sets OPENAI_API_VERSION which langchain_openai requires.
    """
    import os
    provider = (config.llm_provider or "").lower()
    if "azure" in provider:
        if config.llm_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = config.llm_api_key
        azure_endpoint = getattr(config, "azure_endpoint", "") or ""
        if azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
    os.environ.setdefault("OPENAI_API_VERSION", os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"))


def _build_azure_langchain_llm():
    """
    Build an AzureChatOpenAI client for RAGAS / LangChain, correctly handling
    both endpoint formats that may be present in AZURE_OPENAI_ENDPOINT:

      Full URL  (what the .env currently stores):
        https://resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview

      Base URL  (what AzureChatOpenAI actually expects):
        https://resource.cognitiveservices.azure.com/

    This function always normalises to the base URL, extracts the deployment
    and api-version from the URL when the explicit env vars are absent, so
    callers never need to care which format is in the environment.
    """
    import os, re
    from langchain_openai import AzureChatOpenAI  # type: ignore

    raw = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    # Strip to scheme + host so AzureChatOpenAI can append its own path.
    base_match = re.match(r"(https?://[^/]+)", raw)
    base_endpoint = (base_match.group(1) + "/") if base_match else raw

    # Pull deployment from the URL path when AZURE_OPENAI_DEPLOYMENT_NAME is absent.
    dep_match = re.search(r"/deployments/([^/?]+)", raw)
    deployment = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        or (dep_match.group(1) if dep_match else "gpt-4o")
    )

    # Pull api-version from the query string when the env var is absent.
    ver_match = re.search(r"api-version=([^&\s]+)", raw)
    api_version = (
        os.environ.get("OPENAI_API_VERSION")
        or os.environ.get("AZURE_API_VERSION")
        or (ver_match.group(1) if ver_match else "2025-01-01-preview")
    )

    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=api_version,
        azure_endpoint=base_endpoint,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        temperature=0,
        max_tokens=1024,
    )


# ── DeepEval metric helpers ───────────────────────────────────────────────────

def _deepeval_hallucination(
    query: str, response: str, context: list, model
) -> tuple[float, str]:
    """
    DeepEval HallucinationMetric.
    raw score: 0.0 = no hallucination, 1.0 = fully hallucinated.
    stored score: inverted so 1.0 = clean, 0.0 = hallucinated (higher = better).
    Raises on any error — caller records a skipped MetricResult.
    """
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    metric = HallucinationMetric(threshold=0.5, model=model)
    test_case = LLMTestCase(input=query, actual_output=response, context=context[:3])
    metric.measure(test_case)
    raw = metric.score
    if raw is None:
        raise ValueError("HallucinationMetric returned None score")
    # Invert: low raw score (no hallucination) → high stored score (PASS)
    score = round(1.0 - float(raw), 4)
    return score, f"DeepEval HallucinationMetric: {score:.3f} (raw hallucination rate: {float(raw):.3f})"


def _deepeval_answer_relevancy(query: str, response: str, model) -> tuple[float, str]:
    """
    DeepEval AnswerRelevancyMetric.
    Raises on any error — caller records a skipped MetricResult.
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
    DeepEval GEval with a fixed usefulness criterion.
    Raises on any error — caller records a skipped MetricResult.
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


def _deepeval_geval_factual_accuracy(
    query: str, response: str, domain: str, model,
) -> tuple[float, str]:
    """
    Fix 1: GEval-based factual accuracy fallback.
    Used when HallucinationMetric cannot run because no reference context is available
    or expected_behavior is too short (<200 chars) to serve as a reference document.
    Leverages the LLM judge's domain knowledge to detect factual errors without needing
    a ground-truth document. A fitness chatbot recommending dangerous loads, or a medical
    chatbot citing wrong dosages, will score low here.
    Raises on any error — caller records a skipped MetricResult.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="Factual Accuracy",
        criteria=(
            f"The response contains information that is factually sound and appropriate "
            f"for a {domain} AI assistant. It does not make false claims, contradict "
            f"established facts in the {domain} domain, or assert capabilities outside "
            f"the agent's defined scope."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=model,
    )
    test_case = LLMTestCase(input=query, actual_output=response)
    metric.measure(test_case)
    score = round(float(metric.score), 4)
    return score, f"GEval factual accuracy: {score:.3f} (domain: {domain})"


# ── LLM Judge helpers ─────────────────────────────────────────────────────────

async def _llm_judge(
    question: str,
    response: str,
    expected: str,
    llm_client: LLMClient,
    criteria_text: str = _DEFAULT_CRITERIA_TEXT,
) -> dict:
    """
    LLM-as-judge using default criteria only (relevance, accuracy, helpfulness).
    Domain-specific GEval criteria live exclusively in quality.py.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        question=question[:400],
        response=response[:3000],
        expected=expected[:400] if expected else "A helpful, accurate response",
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
        raise


async def _llm_completeness(
    question: str,
    response: str,
    expected: str,
    llm_client: LLMClient,
) -> dict:
    """LLM-based completeness + semantic similarity vs expected_behavior."""
    prompt = COMPLETENESS_PROMPT.format(
        question=question[:400],
        response=response[:2000],
        expected=expected[:600],
    )
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.1, max_tokens=300, task="judge", retries=2,
        )
        return {
            "completeness": float(data.get("completeness", 0.5)),
            "answer_similarity": float(data.get("answer_similarity", 0.5)),
            "reasoning": str(data.get("reasoning", "")),
        }
    except Exception:
        raise


async def _cross_turn_consistency(
    conv: Conversation,
    llm_client: LLMClient,
) -> Optional[tuple[float, str, str]]:
    """
    Fix 6: Check cross-turn consistency for multi-turn conversations.
    Returns (score, reasoning, turns_text) or None if skipped.
    - score: 1.0 = fully consistent, 0.0 = direct contradiction
    - reasoning: judge's explanation (stored in MetricResult.response for verifiability)
    - turns_text: serialized transcript (stored in MetricResult.prompt for verifiability)
    Only meaningful for 2+ turns.
    """
    if len(conv.turns) < 2:
        return None

    lines = []
    for t in conv.turns:
        lines.append(f"Turn {t.turn_number} Q: {t.query[:200]}")
        lines.append(f"Turn {t.turn_number} A: {t.response[:300]}")
    turns_text = "\n".join(lines)

    prompt = CROSS_TURN_CONSISTENCY_PROMPT.format(turns_text=turns_text)
    try:
        data = await llm_client.complete_json(
            prompt, temperature=0.1, max_tokens=200, task="judge", retries=2,
        )
        score     = float(data.get("consistency", 1.0))
        reasoning = str(data.get("reasoning", "") or "No reasoning returned by judge")
        return score, reasoning, turns_text
    except Exception as e:
        raise


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_functional(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,   # kept for signature compat — IGNORED (Fix 3)
) -> List[MetricResult]:
    """
    Evaluate all functional conversations. Returns MetricResult list.

    Fix 3: quality_criteria param is accepted but ignored. LLM Judge always uses
    only the 3 default criteria (relevance, accuracy, helpfulness). Domain-specific
    GEval criteria live exclusively in quality.py to prevent double scoring.
    """
    from core.models import TestClass
    func_convs = [c for c in conversations if c.test_class == TestClass.FUNCTIONAL]
    persona_map = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    _configure_deepeval(config.llm_provider, config.llm_api_key, getattr(config, "azure_endpoint", "") or "")
    deval_model = make_deepeval_model(
        provider_name=config.llm_provider,
        api_key=config.llm_api_key,
        azure_endpoint=getattr(config, "azure_endpoint", "") or "",
    )
    if deval_model is None:
        print(f"[FunctionalEval] WARNING: Could not initialise DeepEval model for provider "
              f"'{config.llm_provider}' — DeepEval metrics (hallucination, answer_relevancy, "
              f"usefulness) will be skipped. Check that the API key for this provider is set.")

    # Fix 3: always default criteria only (no domain criteria from quality_criteria)
    criteria_text  = _DEFAULT_CRITERIA_TEXT
    pass_threshold = THRESHOLDS["functional_pass"]

    # Fix 1: domain used by GEval factual accuracy fallback
    domain = getattr(config, "agent_domain", "general") or "general"

    # Fix 18: which DeepEval metrics to run (driven by config)
    active_deval = set(getattr(config, "deepeval_metrics", ["hallucination", "answer_relevancy"]) or [])

    # Fix 33: enable_judge toggle
    run_judge = getattr(config, "enable_judge", True)

    # sem gates ALL external calls (DeepEval + LLM judge) — max 3 concurrent Azure
    # calls from functional eval at any time, preventing thread-pool exhaustion and
    # Azure rate-limit cascades that cause silent multi-minute freezes.
    sem  = asyncio.Semaphore(3)
    loop = asyncio.get_running_loop()   # Fix 20: was get_event_loop()

    _DEVAL_TIMEOUT = 120  # seconds — hard ceiling per DeepEval call

    async def _eval_one(conv: Conversation) -> List[MetricResult]:
        persona  = persona_map.get(conv.persona_id)
        if not conv.turns:
            return []

        fishbone = persona.fishbone_dimensions if persona else {}
        intent   = persona.intent.value if persona else "genuine"
        local: List[MetricResult] = []

        # ── Per-turn evaluation (Fix 10) ──────────────────────────────────────
        # Evaluate every turn for hallucination + answer_relevancy.
        # Also note the last turn for judge / completeness.
        last_turn = conv.turns[-1]

        for turn in conv.turns:
            query    = turn.query
            response = turn.response
            context  = turn.retrieved_context or []
            expected = turn.expected_behavior or ""

            base_meta = dict(
                conversation_id=conv.conversation_id,
                persona_id=conv.persona_id,
                persona_name=conv.persona_name,
                intent=intent,
                fishbone=fishbone,
                prompt=query,
                response=response[:2000],
                latency_ms=conv.total_latency_ms,
                superset="functional",
                turn_number=turn.turn_number,
            )

            # Error responses — record and skip scoring
            if turn.is_error_response:
                local.append(MetricResult(
                    **base_meta, metric_name="error_response",
                    score=0.0, passed=False,
                    reason=f"Chatbot returned error: {response[:200]}",
                ))
                continue

            # ── Hallucination / Factual Accuracy (Fix 1 + Fix 36) ────────────
            # Branch 1: RAG — use retrieved_context as reference (unchanged)
            # Branch 2: Non-RAG with substantial expected_behavior (>=200 chars) — use as ref
            # Branch 3: No substantial reference — GEval factual accuracy fallback
            #   (uses LLM judge's domain knowledge; no reference document needed)
            if "hallucination" in active_deval and deval_model:
                if context:
                    # Branch 1: RAG — retrieved context chunks as reference
                    try:
                        async with sem:
                            hall_score, hall_reason = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None, _deepeval_hallucination, query, response, context, deval_model,
                                ),
                                timeout=_DEVAL_TIMEOUT,
                            )
                        local.append(MetricResult(
                            **base_meta, metric_name="hallucination",
                            score=hall_score,
                            passed=hall_score >= (1.0 - THRESHOLDS["hallucination_max"]),
                            reason=f"{hall_reason} (ref: retrieved_context)",
                        ))
                    except Exception as e:
                        local.append(MetricResult(
                            **base_meta, metric_name="hallucination",
                            score=0.0, passed=False,
                            reason=f"DeepEval error: {str(e)[:120]}",
                            skipped=True,
                            skip_reason=f"DeepEval error: {str(e)[:120]}",
                        ))

                elif expected and _is_factual_reference(expected):
                    # Branch 2: factual reference document (not a behavioral description)
                    try:
                        async with sem:
                            hall_score, hall_reason = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None, _deepeval_hallucination, query, response, [expected], deval_model,
                                ),
                                timeout=_DEVAL_TIMEOUT,
                            )
                        local.append(MetricResult(
                            **base_meta, metric_name="hallucination",
                            score=hall_score,
                            passed=hall_score >= (1.0 - THRESHOLDS["hallucination_max"]),
                            reason=f"{hall_reason} (ref: expected_behavior)",
                        ))
                    except Exception as e:
                        local.append(MetricResult(
                            **base_meta, metric_name="hallucination",
                            score=0.0, passed=False,
                            reason=f"DeepEval error: {str(e)[:120]}",
                            skipped=True,
                            skip_reason=f"DeepEval error: {str(e)[:120]}",
                        ))

                else:
                    # Branch 3: no usable reference → GEval factual accuracy fallback
                    # Detects domain-level factual errors using the LLM judge's domain knowledge.
                    # Behavioral descriptions (contain directive language like "should", "must")
                    # are not knowledge documents — using them as hallucination references
                    # produces false failures because any valid response diverges from the wording.
                    try:
                        async with sem:
                            fact_score, fact_reason = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None, _deepeval_geval_factual_accuracy, query, response, domain, deval_model,
                                ),
                                timeout=_DEVAL_TIMEOUT,
                            )
                        local.append(MetricResult(
                            **base_meta, metric_name="geval_factual_accuracy",
                            score=fact_score,
                            passed=fact_score >= pass_threshold,
                            reason=fact_reason,
                        ))
                    except Exception as e:
                        local.append(MetricResult(
                            **base_meta, metric_name="geval_factual_accuracy",
                            score=0.0, passed=False,
                            reason=f"GEval error: {str(e)[:120]}",
                            skipped=True,
                            skip_reason=f"GEval error: {str(e)[:120]}",
                        ))

            # ── Answer Relevancy (Fix 18 + Fix 36) ───────────────────────────
            if "answer_relevancy" in active_deval and deval_model:
                try:
                    async with sem:
                        rel_score, rel_reason = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, _deepeval_answer_relevancy, query, response, deval_model,
                            ),
                            timeout=_DEVAL_TIMEOUT,
                        )
                    local.append(MetricResult(
                        **base_meta, metric_name="answer_relevancy",
                        score=rel_score,
                        passed=rel_score >= pass_threshold,
                        reason=rel_reason,
                    ))
                except Exception as e:
                    local.append(MetricResult(
                        **base_meta, metric_name="answer_relevancy",
                        score=0.0, passed=False, reason="",
                        skipped=True,
                        skip_reason=f"DeepEval error: {str(e)[:120]}",
                    ))

        # ── Usefulness — once per conversation (last turn, Fix 36) ───────────
        if deval_model and not last_turn.is_error_response:
            base_last = dict(
                conversation_id=conv.conversation_id,
                persona_id=conv.persona_id,
                persona_name=conv.persona_name,
                intent=intent,
                fishbone=fishbone,
                prompt=last_turn.query[:300],
                response=last_turn.response[:300],
                latency_ms=conv.total_latency_ms,
                superset="functional",
                turn_number=last_turn.turn_number,
            )
            try:
                async with sem:
                    use_score, use_reason = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, _deepeval_usefulness,
                            last_turn.query, last_turn.response, deval_model,
                        ),
                        timeout=_DEVAL_TIMEOUT,
                    )
                local.append(MetricResult(
                    **base_last, metric_name="usefulness",
                    score=use_score,
                    passed=use_score >= pass_threshold,
                    reason=use_reason,
                ))
            except Exception as e:
                local.append(MetricResult(
                    **base_last, metric_name="usefulness",
                    score=0.0, passed=False, reason="",
                    skipped=True,
                    skip_reason=f"DeepEval GEval error: {str(e)[:120]}",
                ))

        # ── LLM Judge (Fix 33: enable_judge guard, Fix 3: default criteria only) ──
        if run_judge and not last_turn.is_error_response:
            base_last = dict(
                conversation_id=conv.conversation_id,
                persona_id=conv.persona_id,
                persona_name=conv.persona_name,
                intent=intent,
                fishbone=fishbone,
                prompt=last_turn.query[:300],
                response=last_turn.response[:300],
                latency_ms=conv.total_latency_ms,
                superset="functional",
                turn_number=last_turn.turn_number,
            )
            async with sem:
                try:
                    judge_result = await _llm_judge(
                        last_turn.query, last_turn.response,
                        last_turn.expected_behavior or "",
                        llm_client, criteria_text,
                    )
                except Exception as e:
                    local.append(MetricResult(
                        **base_last, metric_name="llm_judge",
                        score=0.0, passed=False, reason="",
                        skipped=True,
                        skip_reason=f"LLM judge unavailable: {str(e)[:120]}",
                    ))
                    judge_result = None
            if judge_result is not None:
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
                        **base_last, metric_name=f"llm_{crit_key}", score=s,
                        passed=s >= pass_threshold, reason=reasoning,
                    ))

        # ── Completeness + Answer Similarity (Fix 11) ────────────────────────
        # Only for non-RAG conversations where expected_behavior is available.
        last_expected = last_turn.expected_behavior or ""
        if (
            run_judge
            and not last_turn.is_error_response
            and last_expected
            and not (conv.turns[0].retrieved_context)   # non-RAG
        ):
            base_last = dict(
                conversation_id=conv.conversation_id,
                persona_id=conv.persona_id,
                persona_name=conv.persona_name,
                intent=intent,
                fishbone=fishbone,
                prompt=last_turn.query[:300],
                response=last_turn.response[:300],
                latency_ms=conv.total_latency_ms,
                superset="functional",
                turn_number=last_turn.turn_number,
            )
            async with sem:
                try:
                    comp_result = await _llm_completeness(
                        last_turn.query, last_turn.response, last_expected, llm_client,
                    )
                except Exception as e:
                    for metric_key in ("completeness", "answer_similarity"):
                        local.append(MetricResult(
                            **base_last,
                            metric_name=f"llm_{metric_key}",
                            score=0.0, passed=False, reason="",
                            skipped=True,
                            skip_reason=f"Completeness check unavailable: {str(e)[:120]}",
                        ))
                    comp_result = None
            if comp_result is not None:
                comp_reason = comp_result.get("reasoning", "")
                for metric_key in ("completeness", "answer_similarity"):
                    score = comp_result.get(metric_key, 0.5)
                    local.append(MetricResult(
                        **base_last,
                        metric_name=f"llm_{metric_key}",
                        score=round(score, 4),
                        passed=score >= pass_threshold,
                        reason=comp_reason,
                    ))

        # ── Cross-turn consistency (Fix 10 + Fix 6) ──────────────────────────
        # Fix 6: stores full transcript in prompt, judge reasoning in response
        # so every row is verifiable (no more "[multi-turn: N turns]" placeholder).
        if run_judge and len(conv.turns) >= 2:
            async with sem:
                try:
                    ct_result = await _cross_turn_consistency(conv, llm_client)
                except Exception as e:
                    local.append(MetricResult(
                        conversation_id=conv.conversation_id,
                        persona_id=conv.persona_id,
                        persona_name=conv.persona_name,
                        intent=intent,
                        fishbone=fishbone,
                        prompt=f"[multi-turn: {len(conv.turns)} turns]",
                        response="",
                        latency_ms=conv.total_latency_ms,
                        superset="functional",
                        metric_name="cross_turn_consistency",
                        score=0.0, passed=False, reason="",
                        skipped=True,
                        skip_reason=f"Consistency check unavailable: {str(e)[:120]}",
                    ))
                    ct_result = None
            if ct_result is not None:
                consistency, ct_reasoning, turns_text = ct_result
                local.append(MetricResult(
                    conversation_id=conv.conversation_id,
                    persona_id=conv.persona_id,
                    persona_name=conv.persona_name,
                    intent=intent,
                    fishbone=fishbone,
                    prompt=turns_text[:1000],       # full serialized transcript
                    response=ct_reasoning,           # judge's explanation
                    latency_ms=conv.total_latency_ms,
                    superset="functional",
                    metric_name="cross_turn_consistency",
                    score=round(consistency, 4),
                    passed=consistency >= pass_threshold,
                    reason=f"Cross-turn consistency: {consistency:.3f} — {ct_reasoning}",
                ))

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in func_convs])
    for batch in batches:
        results.extend(batch)
    return results
