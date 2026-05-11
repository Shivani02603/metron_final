"""
Stage 4c: Quality evaluation.
Strict tool-only — no LLM fallback scores.

Metrics:
  - GEval (domain-specific criteria) → DeepEval GEval (Azure OpenAI)
    Skipped per-conversation when:
      * DeepEval throws (rate limit, timeout) — recorded as skipped MetricResult
      * The question is purely factual AND the criterion is policy/procedure-type
        (Fix 2: scope filter prevents false 0.0 scores on factual questions)

RAGAS metrics live exclusively in rag.py (Fix 5: removed from here to prevent
duplicate evaluation with different chunk limits).

Error responses (is_error_response=True) are skipped.
Security conversations (test_class=SECURITY) are skipped — attack prompts trigger
Azure content filter, and domain-quality metrics are not meaningful for adversarial inputs.
"""

from __future__ import annotations
import asyncio
import re
from typing import List, Optional

from core.llm_client import LLMClient
from core.models import (
    Conversation, MetricResult, Persona, RunConfig, TestClass,
)
from core.config import THRESHOLDS
from core.deepeval_azure import make_deepeval_model


# ── Escalation criterion filter ───────────────────────────────────────────────

_ESCALATION_KEYWORDS = frozenset({
    "escalat", "hand off", "handoff", "transfer to human", "human agent",
    "live agent", "escalation appropriateness",
})
_SUPPORT_DOMAINS = frozenset({
    "support", "customer_support", "helpdesk", "help_desk", "triage",
    "service_desk", "contact_center",
})


def _is_escalation_criterion(name: str, description: str) -> bool:
    text = (name + " " + description).lower()
    return any(kw in text for kw in _ESCALATION_KEYWORDS)


def _is_support_agent(config) -> bool:
    domain   = (getattr(config, "agent_domain",      "") or "").lower()
    app_type = (getattr(config, "application_type",  "") or "").lower()
    return any(kw in domain or kw in app_type for kw in _SUPPORT_DOMAINS)


# ── Factual question detector (Fix 2) ────────────────────────────────────────

# Policy/procedure criterion keywords — criteria whose descriptions contain these
# words are not meaningful on purely factual questions.
_POLICY_KEYWORDS = frozenset({
    "policy", "policies", "procedure", "procedures", "compliance", "guideline",
    "guidelines", "regulation", "regulations", "rule", "rules", "protocol",
    "protocols", "standard", "standards", "requirement", "requirements",
})

# Pattern that identifies factual/identity questions (who/what/where/when/how many)
_FACTUAL_PATTERN = re.compile(
    r"^\s*(what|where|who|when|how many|how much|which|is|are|was|were)\b.{0,80}\??\s*$",
    re.IGNORECASE,
)

# Pattern that identifies genuine support/complaint requests where escalation
# criteria are meaningful (issues, errors, complaints, explicit escalation requests).
_SUPPORT_REQUEST_PATTERN = re.compile(
    r"\b(issue|problem|error|broken|not working|crash(?:ing)?|fail(?:ing|ed)?|"
    r"complaint|escalate|escalation|transfer|speak to|talk to|human agent|"
    r"live agent|manager|supervisor|refund|cancel(?:lation)?|support ticket|"
    r"case number|unresolved|ticket|incident)\b",
    re.IGNORECASE,
)


def _is_factual_question(text: str) -> bool:
    """Return True when the query looks like a short factual/identity question."""
    return bool(_FACTUAL_PATTERN.match(text.strip()))


def _is_support_request_query(text: str) -> bool:
    """
    Return True when the query contains language typical of a genuine support or
    complaint scenario (errors, issues, escalation requests, cancellations, etc.).
    Used to gate escalation-appropriateness criteria: those criteria only make
    sense when the user is actually experiencing a problem, not when they are
    asking for drafting help or other non-support tasks.
    """
    return bool(_SUPPORT_REQUEST_PATTERN.search(text.strip()))


def _criterion_is_policy_type(description: str) -> bool:
    """Return True when the criterion description is policy/procedure-oriented."""
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _POLICY_KEYWORDS)


def _should_skip_criterion(query: str, criterion_description: str, criterion_name: str = "") -> bool:
    """
    Fix 2: Skip a domain criterion for this conversation if the question is
    purely factual AND the criterion is policy/procedure-oriented.
    These combinations reliably produce false 0.0 scores.

    Also skips escalation-appropriateness criteria when the query does not
    contain language indicative of a support or complaint scenario.  Escalation
    criteria are only meaningful when a user is reporting an issue or requesting
    human intervention; applying them to e.g. email-drafting or FAQ queries
    produces consistently low scores that do not reflect a real deficiency.
    """
    # Policy-type criterion on a pure factual question
    if _is_factual_question(query) and _criterion_is_policy_type(criterion_description):
        return True
    # Escalation criterion on a non-support query
    if (
        _is_escalation_criterion(criterion_name, criterion_description)
        and not _is_support_request_query(query)
    ):
        return True
    return False


# ── DeepEval GEval ────────────────────────────────────────────────────────────

def _run_deepeval_geval(
    question: str,
    response: str,
    criteria: list,
    criteria_weights: dict,
    model,
) -> dict:
    """
    DeepEval GEval with domain-specific criteria and weighted overall score.

    Fix 4: overall = weighted mean using criterion weights instead of simple average.
    Raises on failure — caller records a skipped MetricResult.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    scores: dict[str, float] = {}
    for criterion in criteria:   # evaluate all configured criteria, not just the first 6
        name        = criterion.get("name", "quality")
        description = criterion.get("description", name)

        # Fix 2: skip policy-type criteria for factual questions,
        # and skip escalation criteria for non-support queries.
        if _should_skip_criterion(question, description, name):
            continue

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
        raise RuntimeError("GEval returned no scores (all criteria skipped or failed)")

    # Fix 4: weighted overall score
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in scores.items():
        w = criteria_weights.get(name, 1.0)
        weighted_sum += score * w
        total_weight  += w
    overall = round(weighted_sum / total_weight, 4) if total_weight > 0 else round(
        sum(scores.values()) / len(scores), 4
    )

    return {"scores": scores, "overall": overall, "method": "deepeval_geval"}


# ── Main evaluator ────────────────────────────────────────────────────────────

async def evaluate_quality(
    conversations: List[Conversation],
    personas: List[Persona],
    config: RunConfig,
    llm_client: LLMClient,
    quality_criteria: Optional[dict] = None,
) -> List[MetricResult]:
    """
    Evaluate quality using DeepEval GEval (domain-specific criteria only).

    Fix 5: RAGAS is NOT run here — it runs exclusively in rag.py.
    Fix 2: Factual questions skip policy-type criteria.
    Fix 4: Criterion weights used in overall score.
    Fix 36: Exceptions produce skipped MetricResult instead of silently dropping.
    """
    from stages.s4_evaluation.functional import _set_azure_env
    _set_azure_env(config)

    deval_model = make_deepeval_model(
        provider_name=config.llm_provider,
        api_key=config.llm_api_key,
        azure_endpoint=getattr(config, "azure_endpoint", "") or "",
    )
    if deval_model is None:
        print(f"[QualityEval] WARNING: Could not initialise DeepEval model for provider "
              f"'{config.llm_provider}' — GEval quality criteria will be skipped. "
              f"Check that the API key for this provider is set.")

    persona_map  = {p.persona_id: p for p in personas}
    results: List[MetricResult] = []

    criteria_list: list = []
    criteria_weights: dict = {}
    if quality_criteria:
        criteria_list    = quality_criteria.get("criteria", [])
        # Build weight map from criteria definitions (default 1.0 if absent)
        criteria_weights = {
            c.get("name", "quality"): float(c.get("weight", 1.0))
            for c in criteria_list
        }

    threshold = (
        quality_criteria.get("passing_threshold", THRESHOLDS["quality_pass"])
        if quality_criteria else THRESHOLDS["quality_pass"]
    )

    sem  = asyncio.Semaphore(3)
    loop = asyncio.get_running_loop()   # Fix 20: was get_event_loop()
    _DEVAL_TIMEOUT = 120  # hard ceiling per GEval call — prevents Azure hang freezing the pipeline

    async def _eval_one(conv: Conversation) -> List[MetricResult]:
        if not conv.turns:
            return []
        last_turn = conv.turns[-1]

        if last_turn.is_error_response:
            return []

        # Skip quality GEval for security conversations
        if conv.test_class == TestClass.SECURITY:
            return []

        persona  = persona_map.get(conv.persona_id)
        fishbone = persona.fishbone_dimensions if persona else {}
        intent   = persona.intent.value if persona else "genuine"

        base_meta = dict(
            conversation_id=conv.conversation_id,
            persona_id=conv.persona_id,
            persona_name=conv.persona_name,
            intent=intent, fishbone=fishbone,
            prompt=last_turn.query,
            response=last_turn.response[:2000],
            latency_ms=conv.total_latency_ms,
            superset="quality",
        )
        local: List[MetricResult] = []

        # ── GEval (domain-specific criteria, Fix 2+4+36) ─────────────────────
        # Respect config.use_geval — the UI toggle that disables GEval entirely.
        # Filter out escalation criteria for non-support agents.
        effective_criteria = [
            c for c in criteria_list
            if _is_support_agent(config) or not _is_escalation_criterion(
                c.get("name", ""), c.get("description", "")
            )
        ]
        if effective_criteria and getattr(config, "use_geval", True):
            async with sem:
                try:
                    geval_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, _run_deepeval_geval,
                            last_turn.query, last_turn.response,
                            effective_criteria, criteria_weights, deval_model
                        ),
                        timeout=_DEVAL_TIMEOUT,
                    )
                    method = geval_result.get("method", "deepeval_geval")
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
                        reason=f"GEval weighted overall (method: {method})",
                    ))
                except Exception as e:
                    # Fix 36: record skipped result instead of silently dropping
                    local.append(MetricResult(
                        **base_meta,
                        metric_name="geval_overall",
                        score=0.0, passed=False, reason="",
                        skipped=True,
                        skip_reason=f"DeepEval GEval error: {str(e)[:120]}",
                    ))

        return local

    batches = await asyncio.gather(*[_eval_one(c) for c in conversations])
    for batch in batches:
        results.extend(batch)
    return results
